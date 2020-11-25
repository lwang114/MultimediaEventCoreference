#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os


# In[2]:


directory ="json"
files = os.listdir(directory)


# In[30]:


import io
results = {}
for file in files:
    with open(os.path.join(directory,file),encoding="utf-8") as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    idx = file.split(".")[0]
    results[idx] = lines


# In[31]:


with open("video_m2e2.json",encoding="utf-8") as f:
    text = f.readlines()
text = json.loads(text[0])


# In[34]:


keys = list(text.keys())
keys.sort()


# In[39]:


with open("short_desc.txt") as f:
    titles = f.readlines()

with open("unannotatedVideos_textEventCount.json") as f:
    unannotated = list(eval(f.readlines()[0]).keys())


# In[7]:


"""
relations = set()
for k, v in results.items():
    for item in v[0]["graph"]["roles"]:
        relations.add(item[2])
"""


# In[8]:


annotation = []
for title in titles:
    idx = text[title[:-1]]["id"].split("=")[1]
    annotation.append(results[idx])
    


# In[78]:


for j in range(len(results)):
    if titles[j][:-1] not in unannotated:
        continue
    text_ex = text[titles[j][:-1]]["long_desc"]
    text_count = 1
    trigger_count = 1
    with open("../brat-v1.3_Crunchy_Frog/data/test/%04d.txt"%j,"w") as f:
        f.write(text_ex)
    with open("../brat-v1.3_Crunchy_Frog/data/test/%04d.ann"%j,"w") as f:
        for example in annotation[j]:
            local_trigger = []
            local_entity = []
            for i in example["graph"]["entities"]:
                for k in range(i[0],i[1]):
                    positions = example["token_ids"][k].split(":")[1].split("-")
                    if k ==i[0]:
                        start = int(positions[0])
                    if k==i[1]-1:
                        end = int(positions[1])+1
                f.write("T%d\t"%text_count + i[2] + " " + str(start)+ " "  +str(end) + "\t"+ text_ex[start:end]+ "\n")
                local_entity.append(text_count)
                text_count += 1
            for i in example["graph"]["triggers"]:
                for k in range(i[0],i[1]):
                    positions = example["token_ids"][k].split(":")[1].split("-")
                    if k ==i[0]:
                        start = int(positions[0])
                    if k==i[1]-1:
                        end = int(positions[1])+1
                f.write("T%d\t"%text_count + i[2].split(".")[1] + " "+ str(start) + " "+ str(end) + "\t" + text_ex[start:end] +"\n")
                local_trigger.append({"id":text_count})
                text_count += 1

            for i in example["graph"]["roles"]:
                key = i[2]
                if key in local_trigger[i[0]]:
                    idx = 2
                    while key + str(idx) in local_trigger[i[0]]:
                        idx+=1
                    key = key + str(idx)
                local_trigger[i[0]][key] = "T%d"%local_entity[i[1]]
            for i,trigger in enumerate(local_trigger):
                f.write("E%d\t"%trigger_count+
                            "%s:T%d"%(example["graph"]["triggers"][i][2].split(".")[1],local_trigger[i]["id"]))
                to_print = []
                for k,v in trigger.items():
                    if k!='id':
                        f.write(" " + k + ":" +v )
                f.write("\n")
                trigger_count += 1


# In[ ]:




