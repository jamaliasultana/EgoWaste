# EgoWaste v0.2_eight_gx_2026-10-16

Local validated export of 8 user-confirmed completed recordings.
Scope: validated_subset_with_quarantine. Unresolved annotations, if any, are listed in quarantine.json.
This is not the full paper benchmark and has not been uploaded.

Each row in annotations.csv is one active-object instance. All paths, including
paths in per-recording CSVs, are relative to this release root. Full RGB frames
are copied once. Lossless binary masks use background=0 and foreground=255.
Boxes are zero-based pixel [x,y,width,height], with exclusive x+width/y+height
edges, derived directly from the released masks. Crops and bbox previews are
convenience images; numeric boxes in annotations.csv are authoritative.

Object tracks are scoped to recording_id. Multiple objects can share a frame.
Category/material IDs are frozen in categories.csv across this release and later
additions; they are not per-video IDs. Source rows are zero-based CSV data-row
positions (header excluded). Repeated equivalent rows were collapsed with an audit.
Review evidence is the user's confirmation that each selected video was reviewed,
not the UI's ambiguous integrated_review_status field.

Every retained source row's event/timestamp/hand metadata is retained in events.json.
When merged rows disagree, that field is blank in annotations.csv and all variants
remain in events.json. Estimated timestamps retain their frame_time_source. This is
source metadata, not a claim that temporal identities or intervals were re-audited.
Train/val/test assignments are unassigned until an explicit benchmark manifest is
provided. Raw videos, audio and transcripts are not included in this frame release.
Dataset licensing and the manuscript's 56-vs-57 taxonomy count must be settled
before publication. Do not treat this local folder as an already licensed release.

Run the supplied exporter with --verify-release PATH to verify every checksum,
image, mask, bounding box, pairing and taxonomy mapping after copying/uploading.
