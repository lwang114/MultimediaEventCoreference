import json

def create_label_dict(mentions):
  label_dict = dict()
  for m in mentions:
    start = min(m['tokens_ids'])
    end = max(m['tokens_ids'])
    if not m['doc_id'] in label_dict:
      label_dict[m['doc_id']] = dict()

    if 'event_type' in m:
      mention_class = m['event_type'].split('.')[-1]

    label_dict[m['doc_id']][(start, end)] = {'cluster_id': m['cluster_id'],
                                             'type': mention_class}
  return label_dict

def convert_video_m2e2_to_vispro(doc_json, 
                                 mention_json, 
                                 video_json, 
                                 ontology_map_json, 
                                 out_file): 
  """
  Returns: a meta data dict of the format (same as VisPros):
    "pronoun_info": a list of dicts of
        "current_pronoun": [int, int] (start and end index, inclusive),
        "candidate_NPs": list of [int, int]'s,
        "reference_type": 0,
        "not_discussed": False, 
    "sentences": list of list of strs,
    "image_file": str,
    "clusters": list of list of [int, int],
    "object_detection": list of ints,
    "doc_key": str
  """
  documents = json.load(open(doc_json))
  text_mentions = json.load(open(mention_json))
  text_mention_dict = create_label_dict(text_mentions)
  video_mention_dict = json.load(open(video_json))
  video_mention_dict = {video_mention_dict[k]['youtube_id']:{"annotations": video_mention_dict[k], 
                                                             "video_file": k} 
                        for k in video_mention_dict}
  
  ontology_map = {visual_type:text_type for text_type, visual_types in json.load(open(ontology_map_json)) for visual_type in visual_types}

  num_docs = 0
  num_coref_docs = 0
  out_f = open(out_file, 'w')
  for doc_id in text_mention_dict:
    # Convert sentences
    sentence_id = -1
    sentences = []
    for token in documents[doc_id]:
      if token[0] != sentence_id:
        sentences.append([])
        sentence_id = token[0]
      sentences[-1].append(token[2])

    # Find overlapping textual and visual labels 
    video_dict = video_mention_dict[doc_id]
    visual_labels = set([ontology_map[a["Event_Type"]] for a in video_dict[doc_id]["annotations"] if a['Event_Type'] in ontology_map])
    text_labels = set([text_mention_dict[doc_id][span]["type"] for span in sorted(text_mention_dict[doc_id])])
    overlapped_labels = visual_labels.intersection(text_labels)
    if len(overlapped_labels) == 0:
      continue

    # Find coreference clusters and pronouns (or mentions of interest)
    clusters = dict()
    pronoun_info = []
    num_mentions = 0
    for span in sorted(text_mention_dict[doc_id]):
      text_dict = text_mention_dict[doc_id][span]
      if text_dict["cluster_id"] > 0:
        if not text_dict["cluster_id"] in clusters:
          clusters[text_dict["cluster_id"]] = []
        clusters[text_dict["cluster_id"]].append(span)
       
       for c in clusters:
         if len(clusters[c]) >= 2:
           num_coref_docs += 1
           break

      if text_dict["type"] in overlapped_labels:
        pronoun = {"current_pronoun": span,
                   "candidate_pronoun": [],
                   "reference_type": 0,
                   "not_discussed": False}
        candidate_pronoun = [p["current_pronoun"] for p in pronoun_info]
        pronoun_info.append(candidate_pronoun)

    doc_dict  = {"pronoun_info": pronoun_info,
                 "sentences": sentences,
                 "image_file": video_dict["video_file"], 
                 "clusters": list(clusters.values()),
                 "object_detection": list(overlapped_labels),
                 "doc_key": doc_id}
    out_f.write(json.dumps(doc_dict)+"\n")
    num_docs += 1
  print(f"Number of documents: {num_docs}, number of documents with coreference: {num_coref_docs}")
