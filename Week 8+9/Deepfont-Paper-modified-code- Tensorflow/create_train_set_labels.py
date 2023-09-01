"""
Author : Rohan Kumar
Date : 10 Jul 2023


Description : Adobe VFR dataset organization

Input : Adobe VFR Dataset
Output : Word attributes classified labels

Filename : create_train_set_labels.py

"""

from itertools import product
import re

def split_string(string):
    pattern = r'[A-Z][a-z]*'
    result = re.findall(pattern, string)
    return result


# Extra attributes that need to be formed
mapper_custom = {
    "smbd":"semibold",
    "ita":"italic",
    "bd":"bold",
    "it":"italic"
}

map_attb = {"none":0,"bold":1,"italic":2,"semibold":3}
map_attb_rev = {0:"none",1:"bold",2:"italic",3:"semibold"}

font_dict={}

classes = {"bold":0,"italic":0,"semibold":0,"none":0,"B+I":0,"SB+I":0}


# write all the search list here
lis_to_search = ["bold","bd","italic","semibold","smbd","ita","it"]

# will find the font_dict and also calculate words that belong to a only particular class in classes
with open('/ssd_scratch/cvit/fontlist.txt','r') as fontlist:
    jai=0

    for num,iterator in enumerate(fontlist):
        font_dict[num]=[]
        useful = iterator.split('-')[-1]
        useful = split_string(useful)
        fg=0
        for key in lis_to_search:
            for i in range(len(useful)):
                if(key==useful[i].lower() and fg==0):
                    try:
                        classes[key]+=1
                        font_dict[num].append(map_attb[key])
                    except:
                        classes[mapper_custom[key]]+=1
                        font_dict[num].append(map_attb[mapper_custom[key]])

                    fg=1
                    break

                elif(key==useful[i].lower() and fg==1):
                    classes[map_attb_rev[font_dict[num][0]]] -=1
                    if map_attb_rev[font_dict[num][0]] !="bold" and map_attb_rev[font_dict[num][0]] !="bd":
                        classes["SB+I"]+=1
                    else:
                        classes["B+I"]+=1
                    try:
                        font_dict[num].append(map_attb[key])

                    except:
                        font_dict[num].append(map_attb[mapper_custom[key]])
                    break

                    fg=2
                
        if fg==2:
            break
        
        if fg==0:
                
            classes["none"]+=1
            font_dict[num].append(0)
