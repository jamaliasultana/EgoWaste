"""Native EgoWaste event, category, video, mask and bbox correction tool.

The tool edits one meta2.csv at a time. It never overwrites source CSVs/masks.
Video clips are played by ffplay so synchronized audio is preserved.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


RAW_ROOT_DEFAULT = Path("/media/jisha/D/trash_dataset/Brookhaven Landfill 20250602")
TRANSCRIPT_ROOT_DEFAULT = Path("/media/jisha/D/trash_dataset/output/Brookhaven Landfill 20250602")
PREPROC_ROOT_DEFAULT = Path("/media/jisha/D/trash_dataset/Brookhaven Landfill 20250602 Preproc")


def text(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def number(v, fallback=0.0):
    try:
        x = float(v)
        return fallback if np.isnan(x) else x
    except Exception:
        return fallback


def parse_xywh(value):
    try:
        vals = ast.literal_eval(text(value)) if isinstance(value, str) else value
        vals = [int(round(float(x))) for x in vals]
        return vals if len(vals) == 4 else None
    except Exception:
        return None


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def collect_full_taxonomy(meta_path, current_df, taxonomy_csv=""):
    """Return all fine categories and their material classes.

    Priority: optional official taxonomy CSV, every meta2.csv under the same
    preprocessed dataset root, and finally the currently opened CSV.
    """
    mapping = {}
    sources = []

    def add_table(table, overwrite=False):
        if "category_name" not in table.columns:
            return
        material_column = next(
            (c for c in ("material_class", "supercategory_name") if c in table.columns),
            None,
        )
        corrected_column = (
            "corrected_category_name" if "corrected_category_name" in table.columns else None
        )
        for _, row in table.iterrows():
            category = text(row.get(corrected_column)) if corrected_column else ""
            category = category or text(row.get("category_name"))
            material = text(row.get(material_column)) if material_column else ""
            if category and category.lower() != "nan":
                if overwrite or category not in mapping:
                    mapping[category] = material

    taxonomy_path = Path(taxonomy_csv).expanduser() if taxonomy_csv else None
    if taxonomy_path and taxonomy_path.is_file():
        table = pd.read_csv(taxonomy_path, low_memory=False)
        add_table(table, overwrite=True)
        sources.append(str(taxonomy_path))

    meta_path = Path(meta_path).resolve()
    # Expected layout: PREPROC_ROOT/Camera N/GX.../data/meta2.csv
    try:
        dataset_root = meta_path.parents[3]
        candidates = sorted(dataset_root.glob("Camera */GX*/data/meta2.csv"))
    except IndexError:
        candidates = []
    for candidate in candidates:
        try:
            table = pd.read_csv(
                candidate,
                usecols=lambda c: c in {
                    "category_name", "corrected_category_name",
                    "supercategory_name", "material_class",
                },
                low_memory=False,
            )
            add_table(table, overwrite=False)
        except Exception:
            continue
    if candidates:
        sources.append(f"{len(candidates)} dataset meta2.csv files")

    add_table(current_df, overwrite=False)
    sources.append("current meta2.csv")
    return mapping, sources


class IntegratedReviewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EgoWaste integrated correction reviewer")
        self.geometry("1580x950")
        self.minsize(1200, 760)

        self.meta_path = None
        self.df = None
        self.filtered_indices = []
        self.position = 0
        self.image_bgr = None
        self.mask = None
        self.original_mask = None
        self.display_scale = 1.0
        self.photo = None
        self.drag_start = None
        self.tool = tk.StringVar(value="add")
        self.brush = tk.IntVar(value=18)
        self.category = tk.StringVar()
        self.material_class = tk.StringVar(value="")
        self.taxonomy_csv = tk.StringVar(value="")
        self.category_to_material = {}
        self.start_time = tk.DoubleVar(value=0)
        self.end_time = tk.DoubleVar(value=0)
        self.note = tk.StringVar()
        self.status = tk.StringVar(value="Open a meta2.csv to begin")
        self.filter_category = tk.StringVar(value="All")
        self.raw_root = tk.StringVar(value=str(RAW_ROOT_DEFAULT))
        self.transcript_root = tk.StringVar(value=str(TRANSCRIPT_ROOT_DEFAULT))
        self.preproc_root = tk.StringVar(value=str(PREPROC_ROOT_DEFAULT))
        self.undo_stack = []
        self.redo_stack = []
        self.events = []
        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=7); top.pack(fill=X)
        ttk.Button(top, text="Open meta2.csv", command=self.open_meta).grid(row=0, column=0, padx=3)
        ttk.Label(top, text="Raw videos").grid(row=0, column=1, sticky="e")
        ttk.Entry(top, textvariable=self.raw_root, width=48).grid(row=0, column=2, sticky="ew")
        ttk.Label(top, text="Transcripts").grid(row=1, column=1, sticky="e")
        ttk.Entry(top, textvariable=self.transcript_root).grid(row=1, column=2, sticky="ew")
        ttk.Label(top, text="Preprocessed frames").grid(row=2, column=1, sticky="e")
        ttk.Entry(top, textvariable=self.preproc_root).grid(row=2, column=2, sticky="ew")
        ttk.Label(top, text="Official taxonomy CSV (optional)").grid(row=3, column=1, sticky="e")
        ttk.Entry(top, textvariable=self.taxonomy_csv).grid(row=3, column=2, sticky="ew")
        ttk.Button(top, text="Choose…", command=self.choose_taxonomy).grid(row=3, column=3, padx=3)
        ttk.Label(top, textvariable=self.status).grid(row=4, column=0, columnspan=4, sticky="w", pady=3)
        top.columnconfigure(2, weight=1)

        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL); pane.pack(fill=BOTH, expand=True, padx=7)
        side = ttk.Frame(pane, padding=5); pane.add(side, weight=1)
        main = ttk.Frame(pane, padding=5); pane.add(main, weight=5)

        ttk.Label(side, text="Filter category").pack(anchor="w")
        self.category_filter_box = ttk.Combobox(side, textvariable=self.filter_category, state="readonly")
        self.category_filter_box.pack(fill=X); self.category_filter_box.bind("<<ComboboxSelected>>", lambda e:self.refresh_filter())
        ttk.Label(side, text="Rows / objects").pack(anchor="w", pady=(8,0))
        lf = ttk.Frame(side); lf.pack(fill=BOTH, expand=True)
        self.rows = tk.Listbox(lf, exportselection=False, width=38)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.rows.yview)
        self.rows.configure(yscrollcommand=sb.set); self.rows.pack(side=LEFT, fill=BOTH, expand=True); sb.pack(side=RIGHT, fill=Y)
        self.rows.bind("<<ListboxSelect>>", self.select_from_list)

        nav = ttk.Frame(main); nav.pack(fill=X)
        ttk.Button(nav, text="← Previous", command=lambda:self.move(-1)).pack(side=LEFT)
        ttk.Button(nav, text="Next →", command=lambda:self.move(1)).pack(side=LEFT, padx=3)
        ttk.Button(nav, text="Play video + audio", command=self.play_video).pack(side=LEFT, padx=12)
        ttk.Button(nav, text="Undo", command=self.undo).pack(side=RIGHT, padx=3)
        ttk.Button(nav, text="Redo", command=self.redo).pack(side=RIGHT, padx=3)
        self.title_label = ttk.Label(nav, text="", font=("TkDefaultFont", 12, "bold")); self.title_label.pack(side=LEFT, padx=12)
        self.detail = ttk.Label(main, text="", wraplength=1100); self.detail.pack(fill=X, pady=4)

        image_frame = ttk.Frame(main); image_frame.pack(fill=BOTH, expand=True)
        self.canvas = tk.Canvas(image_frame, bg="#191919", cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Configure>", lambda e:self.render())

        tools = ttk.LabelFrame(main, text="Mask and bounding box", padding=6); tools.pack(fill=X, pady=4)
        for label, value in [("Add mask", "add"), ("Erase mask", "erase"), ("GrabCut box", "box")]:
            ttk.Radiobutton(tools, text=label, variable=self.tool, value=value).pack(side=LEFT, padx=4)
        ttk.Label(tools, text="Brush").pack(side=LEFT, padx=(15,2))
        ttk.Scale(tools, from_=2, to=80, variable=self.brush, orient=tk.HORIZONTAL, length=130).pack(side=LEFT)
        ttk.Button(tools, text="Reset mask", command=self.reset_mask).pack(side=LEFT, padx=5)
        ttk.Button(tools, text="Clear mask", command=self.clear_mask).pack(side=LEFT, padx=5)
        ttk.Button(tools, text="Save mask + derived bbox", command=self.save_mask).pack(side=RIGHT, padx=5)

        event = ttk.LabelFrame(main, text="Object event and category", padding=7); event.pack(fill=X, pady=4)
        ttk.Label(event, text="Category").grid(row=0,column=0,sticky="w")
        self.category_box = ttk.Combobox(event, textvariable=self.category); self.category_box.grid(row=1,column=0,sticky="ew",padx=3)
        self.category_box.bind("<<ComboboxSelected>>", self.update_material_from_category)
        self.category_box.bind("<FocusOut>", self.update_material_from_category)
        ttk.Label(event, text="Material class").grid(row=0,column=1,sticky="w")
        ttk.Label(event, textvariable=self.material_class, width=18).grid(row=1,column=1,sticky="w",padx=3)
        ttk.Label(event, text="Start time (s)").grid(row=0,column=2,sticky="w")
        ttk.Entry(event, textvariable=self.start_time, width=12).grid(row=1,column=2,padx=3)
        ttk.Button(event,text="Start = current frame",command=self.set_start).grid(row=1,column=3,padx=3)
        ttk.Label(event, text="End time (s)").grid(row=0,column=4,sticky="w")
        ttk.Entry(event, textvariable=self.end_time, width=12).grid(row=1,column=4,padx=3)
        ttk.Button(event,text="End = current frame",command=self.set_end).grid(row=1,column=5,padx=3)
        ttk.Label(event,text="Note").grid(row=0,column=6,sticky="w")
        ttk.Entry(event,textvariable=self.note).grid(row=1,column=6,sticky="ew",padx=3)
        ttk.Button(event,text="Save/update this object event",command=self.save_event).grid(row=2,column=0,columnspan=2,sticky="ew",pady=5)
        ttk.Button(event,text="New object starts here",command=self.new_object_here).grid(row=2,column=2,columnspan=2,sticky="ew",padx=4)
        ttk.Button(event,text="Delete/false object",command=self.delete_object).grid(row=2,column=4,sticky="ew",padx=4)
        ttk.Button(event,text="Export corrected meta2.csv",command=self.export).grid(row=2,column=5,columnspan=2,sticky="ew",padx=4)
        event.columnconfigure(0,weight=2); event.columnconfigure(6,weight=2)

    def choose_taxonomy(self):
        path = filedialog.askopenfilename(
            title="Choose official EgoWaste taxonomy CSV",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            self.taxonomy_csv.set(path)

    def update_material_from_category(self, _event=None):
        category = self.category.get().strip()
        material = self.category_to_material.get(category, "")
        self.material_class.set(material or "Unknown—check taxonomy")

    def open_meta(self):
        p=filedialog.askopenfilename(title="Open one meta2.csv",filetypes=[("CSV","*.csv")])
        if not p:return
        try:
            selected_path=Path(p).resolve()
            # Users normally select the immutable original meta2.csv. If a
            # working export exists, resume it so previous corrections remain
            # visible. Also support selecting the corrected CSV directly.
            if selected_path.parent.name == "integrated_corrections":
                self.output_dir=selected_path.parent
                original_candidate=self.output_dir.parent/"meta2.csv"
                self.meta_path=original_candidate if original_candidate.is_file() else selected_path
                working_path=selected_path
                resumed=True
            else:
                self.meta_path=selected_path
                self.output_dir=self.meta_path.parent/"integrated_corrections"
                self.output_dir.mkdir(parents=True,exist_ok=True)
                working_path=self.output_dir/"meta2_integrated_corrected.csv"
                resumed=False
                if working_path.is_file():
                    resumed=messagebox.askyesno(
                        "Resume saved corrections?",
                        "Saved corrections were found:\n\n"
                        f"{working_path}\n\n"
                        "Choose Yes to continue from them. Choose No only to "
                        "restart from the original meta2.csv."
                    )
                if not resumed:
                    working_path=self.meta_path

            df=pd.read_csv(working_path,low_memory=False)
            required={"camera_id","gx_id","category_name","frame_name","object_track_id","bbox_xywh"}
            missing=required-set(df.columns)
            if missing:raise ValueError("Missing columns: "+", ".join(sorted(missing)))
            self.df=df
            for c,d in [("corrected_category_name",df["category_name"]),("corrected_mask_path",""),("corrected_bbox_xywh",df["bbox_xywh"]),("manual_event_start",df.get("begin_time",pd.Series(np.nan,index=df.index))),("manual_event_end",df.get("end_time",pd.Series(np.nan,index=df.index))),("integrated_review_status","unreviewed"),("integrated_review_note","")]:
                if c not in df:df[c]=d
            self.output_dir.mkdir(parents=True,exist_ok=True)
            self.working_csv_path=self.output_dir/"meta2_integrated_corrected.csv"
            self.audit_path=self.output_dir/"audit_log.jsonl"
            self.event_path=self.output_dir/"object_events.json"
            if self.event_path.exists():
                self.events=json.loads(self.event_path.read_text())
            self.category_to_material, taxonomy_sources = collect_full_taxonomy(
                self.meta_path, df, self.taxonomy_csv.get()
            )
            categories=sorted(self.category_to_material, key=str.casefold)
            self.category_box["values"]=categories; self.category_filter_box["values"]=["All"]+categories
            self.filter_category.set("All"); self.refresh_filter()
            self.status.set(
                f"{'Resumed saved corrections' if resumed else 'Loaded original data'}: "
                f"{len(df):,} rows and {len(categories)} classes from "
                f"{', '.join(taxonomy_sources)}"
            )
        except Exception as e:messagebox.showerror("Cannot open CSV",str(e))

    def autosave_working_copy(self):
        """Atomically save current metadata and events after every committed action."""
        if self.df is None:return
        self.output_dir.mkdir(parents=True,exist_ok=True)
        output=getattr(
            self,"working_csv_path",
            self.output_dir/"meta2_integrated_corrected.csv"
        )
        temporary=output.with_suffix(".csv.tmp")
        self.df.to_csv(temporary,index=False)
        temporary.replace(output)
        event_temporary=self.event_path.with_suffix(".json.tmp")
        event_temporary.write_text(json.dumps(self.events,indent=2))
        event_temporary.replace(self.event_path)

    def refresh_filter(self):
        if self.df is None:return
        cat=self.filter_category.get()
        if cat=="All":inds=list(self.df.index)
        else:inds=list(self.df.index[self.df["corrected_category_name"].fillna(self.df["category_name"]).astype(str)==cat])
        self.filtered_indices=inds; self.rows.delete(0,END)
        for i in inds:
            r=self.df.loc[i]
            self.rows.insert(END,f"{r.gx_id} · f{int(r.frame_name):06d} · obj {r.object_track_id} · {text(r.corrected_category_name) or r.category_name}")
        self.position=0
        if inds:self.rows.selection_set(0);self.load_current()

    def select_from_list(self,_=None):
        s=self.rows.curselection()
        if s:self.position=s[0];self.load_current()

    def move(self,delta):
        if not self.filtered_indices:return
        self.position=max(0,min(len(self.filtered_indices)-1,self.position+delta))
        self.rows.selection_clear(0,END);self.rows.selection_set(self.position);self.rows.see(self.position);self.load_current()

    @property
    def idx(self):return self.filtered_indices[self.position]
    @property
    def row(self):return self.df.loc[self.idx]

    def frame_path(self,r):
        candidates=[Path(text(r.get("source_image_path"))),Path(text(r.get("image_path"))),Path(self.preproc_root.get())/f"Camera {int(r.camera_id)}"/text(r.gx_id)/"frames_resized_1024"/f"{int(r.frame_name):06d}.jpg"]
        return next((p for p in candidates if str(p) and p.is_file()),None)

    def mask_path(self,r):
        candidates=[Path(text(r.get("corrected_mask_path"))),Path(text(r.get("mask_path"))),Path(text(r.get("source_mask_path")))]
        return next((p for p in candidates if str(p) and p.is_file()),None)

    def load_current(self):
        r=self.row;p=self.frame_path(r)
        if not p:
            self.status.set(f"Missing frame for row {self.idx}");return
        self.image_bgr=cv2.imread(str(p));mp=self.mask_path(r)
        if mp:
            m=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
            self.mask=cv2.resize(m,(self.image_bgr.shape[1],self.image_bgr.shape[0]),interpolation=cv2.INTER_NEAREST)>0 if m is not None else np.zeros(self.image_bgr.shape[:2],bool)
        else:self.mask=np.zeros(self.image_bgr.shape[:2],bool)
        self.original_mask=self.mask.copy();self.undo_stack=[];self.redo_stack=[]
        self.category.set(text(r.corrected_category_name) or text(r.category_name))
        self.update_material_from_category()
        cur=number(r.get("frame_time_seconds"),number(r.get("begin_time"),0))
        self.start_time.set(number(r.get("manual_event_start"),number(r.get("begin_time"),cur)))
        self.end_time.set(number(r.get("manual_event_end"),number(r.get("end_time"),cur)))
        self.note.set(text(r.get("integrated_review_note")))
        self.title_label.config(text=f"{r.gx_id} · frame {int(r.frame_name):06d} · object {r.object_track_id}")
        self.detail.config(text=f"Phrase: {text(r.get('phrase'))}   Frame time: {cur:.3f}s   Existing interval: {text(r.get('begin_time'))}–{text(r.get('end_time'))}s   Hand: {text(r.get('hand_side'))}")
        self.render()

    def render(self):
        if self.image_bgr is None or self.canvas.winfo_width()<20:return
        rgb=cv2.cvtColor(self.image_bgr,cv2.COLOR_BGR2RGB); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); out=np.dstack([gray]*3)
        out=(out*.45).astype(np.uint8);out[self.mask]=rgb[self.mask]
        # green boundary and bbox derived from current mask
        contours,_=cv2.findContours(self.mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out,contours,-1,(0,255,60),2)
        box=mask_bbox(self.mask)
        if box:
            x,y,w,h=box;cv2.rectangle(out,(x,y),(x+w-1,y+h-1),(255,210,0),2)
        h,w=out.shape[:2];scale=min(self.canvas.winfo_width()/w,max(100,self.canvas.winfo_height())/h,1.0)
        scale=min(scale,1.25);self.display_scale=scale
        shown=cv2.resize(out,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        self.photo=ImageTk.PhotoImage(Image.fromarray(shown));self.canvas.delete("all");self.canvas.create_image(0,0,image=self.photo,anchor="nw")

    def xy(self,e):return int(e.x/self.display_scale),int(e.y/self.display_scale)
    def snapshot(self):self.undo_stack.append(self.mask.copy());self.redo_stack.clear()
    def mouse_down(self,e):
        if self.mask is None:return
        self.snapshot();self.drag_start=self.xy(e)
        if self.tool.get()!="box":self.paint(*self.drag_start)
    def mouse_drag(self,e):
        if self.tool.get()!="box":self.paint(*self.xy(e))
    def mouse_up(self,e):
        if self.tool.get()=="box" and self.drag_start:
            x0,y0=self.drag_start;x1,y1=self.xy(e);x0,x1=sorted((max(0,x0),min(self.mask.shape[1]-1,x1)));y0,y1=sorted((max(0,y0),min(self.mask.shape[0]-1,y1)))
            if x1-x0>3 and y1-y0>3:
                gc=np.zeros(self.mask.shape,np.uint8);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
                try:
                    cv2.grabCut(self.image_bgr,gc,(x0,y0,x1-x0,y1-y0),bg,fg,5,cv2.GC_INIT_WITH_RECT);self.mask=np.isin(gc,[cv2.GC_FGD,cv2.GC_PR_FGD])
                except cv2.error:pass
        self.drag_start=None;self.render()
    def paint(self,x,y):
        if 0<=x<self.mask.shape[1] and 0<=y<self.mask.shape[0]:cv2.circle(self.mask,(x,y),int(self.brush.get()),self.tool.get()=="add",-1);self.render()
    def reset_mask(self):self.snapshot();self.mask=self.original_mask.copy();self.render()
    def clear_mask(self):self.snapshot();self.mask[:]=False;self.render()
    def undo(self):
        if self.undo_stack:self.redo_stack.append(self.mask.copy());self.mask=self.undo_stack.pop();self.render()
    def redo(self):
        if self.redo_stack:self.undo_stack.append(self.mask.copy());self.mask=self.redo_stack.pop();self.render()

    # def video_path(self,r):return Path(self.raw_root.get())/f"Camera{int(r.camera_id)}"/f"{text(r.gx_id)}.MP4"
    def video_path(self, r):
        raw_root = Path(self.raw_root.get()).expanduser()
        camera_id = int(float(r.camera_id))
        gx_id = text(r.gx_id).strip()

        # Support different camera-folder naming conventions.
        camera_folder_names = [
            f"Camera {camera_id}",   # Camera 1
            f"Camera{camera_id}",    # Camera1
            f"camera {camera_id}",   # camera 1
            f"camera{camera_id}",    # camera1
        ]

        video_extensions = {
            ".mp4",
            ".mov",
            ".avi",
            ".m4v",
        }

        # First, try the expected GX filename with common extensions.
        for folder_name in camera_folder_names:
            camera_folder = raw_root / folder_name

            for extension in [
                ".MP4",
                ".mp4",
                ".MOV",
                ".mov",
                ".AVI",
                ".avi",
                ".M4V",
                ".m4v",
            ]:
                candidate = camera_folder / f"{gx_id}{extension}"

                if candidate.is_file():
                    return candidate

        # Fallback: search case-insensitively within each possible camera folder.
        for folder_name in camera_folder_names:
            camera_folder = raw_root / folder_name

            if not camera_folder.is_dir():
                continue

            for candidate in camera_folder.iterdir():
                if not candidate.is_file():
                    continue

                if (
                    candidate.stem.casefold() == gx_id.casefold()
                    and candidate.suffix.casefold()
                    in video_extensions
                ):
                    return candidate

        # Return the most likely path so the missing-video error is informative.
        return (
            raw_root
            / f"Camera {camera_id}"
            / f"{gx_id}.MP4"
        )
    
    def play_video(self):
        if self.df is None:return
        p=self.video_path(self.row)
        if not p.is_file():messagebox.showerror("Video missing",str(p));return
        start=max(0,self.start_time.get()-1);duration=max(2,self.end_time.get()-self.start_time.get()+2)
        player=shutil.which("ffplay")
        if not player:messagebox.showerror("ffplay missing","Install FFmpeg: sudo apt install ffmpeg");return
        subprocess.Popen([player,"-autoexit","-ss",f"{start:.3f}","-t",f"{duration:.3f}",str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def current_time(self):
        r=self.row
        direct=number(r.get("frame_time_seconds"),float("nan"))
        if not np.isnan(direct):return direct
        direct=number(r.get("begin_time"),float("nan"))
        if not np.isnan(direct):return direct
        # Recovered rows can lack timestamps. Interpolate from timestamped rows
        # in the same GX using extracted frame index as the coordinate.
        same=self.df[(self.df["gx_id"].astype(str)==text(r.gx_id)) & self.df["frame_time_seconds"].notna()]
        if len(same)>=2:
            known=same[["frame_name","frame_time_seconds"]].drop_duplicates("frame_name").sort_values("frame_name")
            return float(np.interp(float(r.frame_name),known["frame_name"].astype(float),known["frame_time_seconds"].astype(float)))
        return 0.0
    def set_start(self):self.start_time.set(self.current_time())
    def set_end(self):self.end_time.set(self.current_time())
    def audit(self,action,details):
        record={"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"action":action,"meta2":str(self.meta_path),"row":int(self.idx),"gx_id":text(self.row.gx_id),"frame_name":int(self.row.frame_name),"object_track_id":text(self.row.object_track_id),"details":details}
        with self.audit_path.open("a") as f:f.write(json.dumps(record)+"\n")

    def save_mask(self):
        if self.mask is None:return
        r=self.row;folder=self.output_dir/"masks"/text(r.gx_id);folder.mkdir(parents=True,exist_ok=True)
        p=folder/f"{int(r.frame_name):06d}_objG{int(float(r.object_track_id)):05d}.png";cv2.imwrite(str(p),self.mask.astype(np.uint8)*255)
        box=mask_bbox(self.mask)
        self.df.at[self.idx,"corrected_mask_path"]=str(p);self.df.at[self.idx,"corrected_bbox_xywh"]=json.dumps(box) if box else ""
        self.df.at[self.idx,"integrated_review_status"]="mask_corrected";self.audit("save_mask",{"mask":str(p),"bbox_xywh":box})
        self.autosave_working_copy()
        self.status.set(f"Saved corrected mask and bbox: {p}")

    def save_event(self):
        if self.df is None:return
        s,e=float(self.start_time.get()),float(self.end_time.get())
        if e<s:messagebox.showerror("Invalid interval","End time must be greater than or equal to start time.");return
        selected_category=self.category.get().strip()
        if not selected_category:
            messagebox.showerror("Missing category", "Choose or type a category before saving.");return
        if selected_category not in self.category_to_material:
            if not messagebox.askyesno(
                "Category not in taxonomy",
                f"{selected_category!r} is not in the loaded taxonomy. Save it anyway?"
            ):return
        selected_material=self.category_to_material.get(selected_category, text(self.row.get("supercategory_name")))
        r=self.row;event={"event_id":f"{r.gx_id}_obj{r.object_track_id}_{s:.3f}","gx_id":text(r.gx_id),"camera_id":int(r.camera_id),"source_row":int(self.idx),"object_track_id":text(r.object_track_id),"start_time":s,"end_time":e,"category_name":selected_category,"material_class":selected_material,"hand_side":text(r.get("hand_side")),"phrase":text(r.get("phrase")),"note":self.note.get()}
        self.events=[x for x in self.events if x["event_id"]!=event["event_id"]];self.events.append(event);self.event_path.write_text(json.dumps(self.events,indent=2))
        self.df.at[self.idx,"corrected_category_name"]=event["category_name"]
        self.df.at[self.idx,"supercategory_name"]=selected_material
        self.df.at[self.idx,"corrected_supercategory_name"]=selected_material
        self.df.at[self.idx,"manual_event_start"]=s;self.df.at[self.idx,"manual_event_end"]=e;self.df.at[self.idx,"integrated_review_note"]=self.note.get();self.df.at[self.idx,"integrated_review_status"]="event_corrected"
        self.audit("save_event",event);self.autosave_working_copy();self.status.set("Saved object event correction (autosaved)")

    def new_object_here(self):
        self.save_event();now=self.current_time()
        # Duplicate the current temporal row as a new, manually created physical
        # object. Its mask begins empty and receives a new track ID.
        new=self.row.copy();numeric=pd.to_numeric(self.df["object_track_id"],errors="coerce")
        new_track=int(numeric.max())+1 if numeric.notna().any() else len(self.df)+1
        new["object_track_id"]=new_track;new["corrected_category_name"]=""
        new["corrected_mask_path"]="";new["corrected_bbox_xywh"]=""
        new["manual_event_start"]=now;new["manual_event_end"]=max(now,number(self.row.get("end_time"),now))
        new["integrated_review_status"]="manual_new_object";new["integrated_review_note"]="New object split from previous event"
        new_index=int(self.df.index.max())+1;self.df.loc[new_index]=new
        self.audit("create_new_object",{"new_source_row":new_index,"new_object_track_id":new_track,"start_time":now})
        self.autosave_working_copy()
        self.filter_category.set("All");self.refresh_filter();self.position=self.filtered_indices.index(new_index)
        self.rows.selection_clear(0,END);self.rows.selection_set(self.position);self.rows.see(self.position);self.load_current()
        self.mask[:]=False;self.original_mask=self.mask.copy();self.category.set("");self.note.set("New object split from previous event");self.render()
        messagebox.showinfo("New object created",f"Created manual object track {new_track} at {now:.3f}s. Choose its category, adjust the end time, draw/GrabCut its mask, save the mask, then save the event.")
    def delete_object(self):
        if messagebox.askyesno("Mark false object","Mark this row/object as deleted or false?"):
            self.df.at[self.idx,"integrated_review_status"]="deleted_false_object";self.audit("delete_object",{"reason":self.note.get()});self.autosave_working_copy();self.status.set("Marked as false/deleted and autosaved")

    def export(self):
        if self.df is None:return
        self.autosave_working_copy();out=self.working_csv_path;messagebox.showinfo("Export complete",f"Corrected CSV:\n{out}\n\nOriginal meta2.csv was not changed.")


if __name__=="__main__":IntegratedReviewer().mainloop()
