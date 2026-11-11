"""Select unprocessed uploads, including projects created before processing markers."""
def pending_uploads(state):
    processed = set(state.get('processed_uploads', []))
    uploads = state.get('uploads', [])
    for clip in state.get('clips', []):
        if clip.get('source_file'):
            processed.add(clip['source_file'])
        else:
            matches = [u for u in uploads if u['name'] == clip.get('source')]
            if len(matches) == 1:
                processed.add(matches[0]['file'])
    return [u for u in uploads if u['file'] not in processed]
