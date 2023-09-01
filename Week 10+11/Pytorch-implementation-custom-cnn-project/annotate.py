import numpy as np
import json
import cv2

with open('/scratch/bbox_info_res.json','r') as file:
    bbox = json.load(file)

for image_file in bbox.keys():
    img = cv2.imread(f'../test_corpus/{image_file}')
    for dim in bbox[image_file]:
        bold_votes = 0
        italic_votes=0
        for id_dict in dim["bb_ids"]:
            id_dict = id_dict["attb"]
            bold_votes+=int(id_dict["isBold"])
            italic_votes+=int(id_dict["isItalic"])
        
        if len(dim["bb_ids"])>1:
            print("dhokha")
        else:
            x,y,z,w = dim["bb_dim"]
            fg=0
            if bold_votes+italic_votes==2:
                print("hi")
                cv2.rectangle(img,(x,y),(z,w),(0,255,0),2)
            
            elif bold_votes==1:
                # print("hello")
                cv2.rectangle(img,(x,y),(z,w),(0,0,255),2)

            elif italic_votes==1:
                print("-------------------------------------")
                cv2.rectangle(img,(x,y),(z,w),(255,0,0),2)
    print(img.shape)
    cv2.imwrite(f'./modified_output/{image_file}',img)
        
   


