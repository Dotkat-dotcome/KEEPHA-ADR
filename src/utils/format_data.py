import os, sys
import random
import re
from collections import Counter
import shutil
import itertools
import argparse

rel_condition = [["DRUG","DISORDER"],
    ["DRUG","FUNCTION"],
    ["DISORDER","DISORDER"],
    ["DISORDER","FUNCTION"],
    ["DISORDER","ANATOMY"],
    ["DRUG","MEASURE"],
    ["DRUG","TIME"],
    ["DISORDER","TIME"],
    ["CHANGE_TRIGGER","DRUG"]]

def overlap(x, y):
        a = max(x.spans[0][0], y.spans[0][0])
        b = min(x.spans[0][1], y.spans[0][1])
        return bool(b-a > 0)

class Term:
    def __init__(self,id,name,spans,text):
        self.id = id
        self.name = name
        self.spans = sorted(spans, key=lambda x:x[0]) # list of list
        self.text = text

class Attribute:
    def __init__(self, id, type,reference,text):
        self.id = id
        self.type = type
        self.reference = reference
        self.text = text

class Relation:
    def __init__(self, id, name, argnames, references):
        self.id = id
        self.name = name
        self.argnames = argnames
        self.references = references



class ParsedObject:
    def __init__(self,t,data):
        self.type=t
        self.data=data


class AnnotationFile:
    def __init__(self, filename, terms=None, attributes=None, relations=None):
        if filename:
            self.create_from_file(filename)
        else:
            self.create_from_prediction(terms, attributes, relations)


    def parse_term(self,line):
        t="Term"
        i0 = line.find("\t")
        id = line[0:i0]
        i1 = line[i0+1:].find("\t")

        iName = line[i0+1:].find(" ")
        name = line[i0+1:i0+1+iName]

        span_txt = line[i0+1+iName+1:i1+i0+1]
        text = line[i1+i0+1+1:]
        spans = span_txt.split(";")
        for i in range(len(spans)):
            tmp = spans[i].split()
            spans[i] = [0,0]
            spans[i][0] = int(tmp[0])
            spans[i][1] = int(tmp[1])

        return ParsedObject(t, Term(id,name,spans,text))

    def parse_attr(self,line):
        t = "Attribute"
        words = line.split()
        if len(words)==4:
            id = words[0]
            type = words[1]
            reference = words[2]
            name = words[3]
        elif len(words)==3: # Negation has no name
            id = words[0]
            type = words[1]
            reference = words[2]
            name = ''
        else:
            print("PARSING ERROR: ATT FORMAT EXCEPTION")
        return ParsedObject(t,Attribute(id,type,reference,name))

    def parse_relation(self,line):
        t = "Relation"
        words = line.split()
        id = words[0]
        name = words[1]
        argnames = []
        args = []
        for i in range(2,len(words)):
            tmp = words[i].split(":")
            argnames += [tmp[0]]
            args+=[tmp[1]]

        return ParsedObject(t, Relation(id,name,argnames,args))



    def parse_line(self,line):
        if(line.startswith("T")):
            return self.parse_term(line)
        elif(line.startswith("A")):
            return self.parse_attr(line)
        elif(line.startswith("R")):
            return self.parse_relation(line)
        else:
            raise RuntimeError()


    def create_from_prediction(self, terms, attributes, relations):
        self.terms = {}
        self.attributes = {}
        self.relations = {}
        
        for term in terms:
            self.terms[term.id] = term
        for attribute in attributes:
            self.attributes[attribute.id] = attribute
        for relation in relations:
            self.relations[relation.id] = relation
        

    def create_from_file(self,filename):
        self.terms = {}
        self.attributes = {}
        self.relations = {}
        lines = open(filename).readlines()

        for line in lines:
            a = self.parse_line(line)
            if(a.type=="Term"):
                self.terms[a.data.id]=a.data
            elif(a.type=="Attribute"):
                self.attributes[a.data.id]=a.data
            else:
                self.relations[a.data.id]=a.data

        self.relationsByReference = {}
        self.attributesByReference = {}
        for t in self.terms.values():
            self.relationsByReference[t.id] = []
            self.attributesByReference[t.id] = []

        for r in self.relations.values():
            for t in r.references:
                self.relationsByReference[t] += [r]

        for r in self.attributes.values():
                self.attributesByReference[r.reference] += [r]

    def write(self,filename):
        f = open(filename,"w")
        for t in self.terms.values():
            txt = ""
            txt+= t.id
            txt+="\t"
            txt+=t.name
            txt+= " "
            for k in range(len(t.spans)):
                txt+=str(t.spans[k][0])
                txt+=" "
                txt+=str(t.spans[k][1])
                if(k<len(t.spans)-1):
                    txt+=";"
            txt+="\t"
            txt+=t.text
            print(txt,file=f,end="") # end="" for creating dataset, w/o for writing the prediction
        for a in self.attributes.values():
            txt=""
            txt+=a.id
            txt+="\t"
            txt+=a.type
            txt+=" "
            txt+=a.reference
            txt+=" "
            txt+=a.text
            print(txt,file=f)
        for r in self.relations.values():
            txt=""
            txt+=r.id
            txt+="\t"
            txt+=r.name
            for i in range(len(r.references)):
                txt+=" "
                txt+=r.argnames[i]+":"+r.references[i]
            
            
            print(txt,file=f)

    def getBadTerms(self):
        badTerms = []
        for t in self.terms.values():
            if(len(t.spans)>1):
                badTerms+=[t]
        a = self.terms.values()
        a = sorted(a, key=lambda x: x.spans[0][0])
        added_badTerms = [False] * len(a)

        i=0 
        while i < len(a):
            if len(a[i].spans)>1:
                i+=1
                continue
            j=i+1
            while j < len(a):
                if len(a[j].spans)>1:
                    j+=1
                    continue

                if overlap(a[i], a[j]):
                    if not added_badTerms[i]:
                        added_badTerms[i] = True
                        badTerms+=[a[i]]
                    if not added_badTerms[j]:
                        added_badTerms[j] = True
                        badTerms+=[a[j]]
                    j+=1
                else:
                    break
                    # after sorting, if not overlapped then what comes later would not overlap either.
            i+=1

        return badTerms



    def removeTerm(self,term_id):
        del self.terms[term_id]
        for r in self.relationsByReference[term_id]:
            if r.id in self.relations:
                del self.relations[r.id]
        for r in self.attributesByReference[term_id]:
            if r.id in self.attributes:
                del self.attributes[r.id]

        del self.relationsByReference[term_id]
        del self.attributesByReference[term_id]


    def deleteSpans(self,spans):
        #NOTE: spans do not overlap with any other references
        #we assume terms do not range over a sentence

        #delete all terms and relations and attributes
        #in the spans

        #assumption: our spans go until the end of the sentence
        #
        terms_in_spans =[]
        terms = self.terms.values()
        terms = sorted(terms,key=lambda x: x.spans[0][0])
        i = 0

        for s in spans:
            while terms[i].spans[0][0] < s[0]:
                i+=1 
            while(i<len(terms) and terms[i].spans[0][0] < s[1] and terms[i].spans[0][0]>=s[0]):
                terms_in_spans+=[terms[i].id]
                i+=1
        
        for t in terms_in_spans:
            self.removeTerm(t)
        
        terms = self.terms.values()
        terms = sorted(terms,key=lambda x: x.spans[0][0])

        #WE ASSUME TERMS ONLY HAVE A SINGLE SPAN NOW
        i=0
        to_remove=0
        for t in terms:
            while i!=len(spans) and t.spans[0][0] >= spans[i][1]:
                to_remove+=spans[i][1]-spans[i][0]
                i+=1
            t.spans[0][0]-=to_remove
            t.spans[0][1]-=to_remove
        


def mergeSpans(spans):
    spans = sorted(spans,key=lambda x: x[0])
    new_spans = []
    new_spans +=[spans[0]]
    for i in range(1,len(spans)):
        if(new_spans[-1][1]>=spans[i][0]):
            new_spans[-1][1]=max(new_spans[-1][1],spans[i][1])
        else:
            new_spans+=[spans[i]]
    return new_spans

def getFullSentences(text,spans):

    text2 = text[:]
    text2 = text2.replace('．', '.')
    text2 = text2.replace('。', '.')
    text2 = text2.replace('\n', '.')
    for s in spans:
        id0 = text2[:s[0]].rfind(".")
        id1 = text2[s[1]:].find(".")
        if id0<0:
            id0 =0
        else:
            id0+=1
        if(id1<0):
            id1 = len(text)
        else:
            id1 = id1+1

        s[0]=id0
        s[1]=id1+s[1]

    return spans

def removeSpans(text,spans):
    for s in reversed(spans):
        text=text[:s[0]]+text[s[1]:]
    return text


def fixAnnotations(ANNOTATION_FILE,
                   TEXT_FILE,
                   ANNOTATION_FILE_OUTPUT,
                   TEXT_FILE_OUTPUT):

    a = AnnotationFile(ANNOTATION_FILE)
    print(ANNOTATION_FILE)
    badTerms = a.getBadTerms()

    if len(badTerms) == 0:
        shutil.copy(ANNOTATION_FILE, ANNOTATION_FILE_OUTPUT)
        shutil.copy(TEXT_FILE, TEXT_FILE_OUTPUT)

        return None

    else:
        spans = []
        for t in badTerms:
            spans += [[t.spans[0][0], t.spans[-1][1]]]

        text = open(TEXT_FILE).read()
        spans = mergeSpans(spans)
        spans = getFullSentences(text,spans)
        spans = mergeSpans(spans)
        #they are sorted after merging

        out_text = open(TEXT_FILE_OUTPUT,"w")
 
        text = removeSpans(text,spans)
        out_text.write(text)

        #removed spans
        a.deleteSpans(spans)
        a.write(ANNOTATION_FILE_OUTPUT)

        return ANNOTATION_FILE.split("/")[-1]


def addNoneAnnotations(ANNOTATION_FILE,
                   TEXT_FILE,
                   ANNOTATION_FILE_OUTPUT,
                   TEXT_FILE_OUTPUT):

    a = AnnotationFile(ANNOTATION_FILE)
    print(ANNOTATION_FILE)
    print("hihi")

    badTerms_init = a.getBadTerms()

    # already-linked pairs
    linked_pairs = [] 
    for rel in a.relations:
        linked_pairs.append(a.relations[rel].references)

    # get pairs
    pairs = []
    for i, j in itertools.product(a.terms, a.terms):
        # not self-link
        if i==j:
            pass
        # satify entity-type conditions
        if [a.terms[i].name, a.terms[j].name] in rel_condition:
            # not already-linked
            if [a.terms[i].name, a.terms[j].name] not in linked_pairs:
                # not badTerms
                if a.terms[i] not in badTerms_init and a.terms[j] not in badTerms_init:
                    pairs.append([a.terms[i].id, a.terms[j].id])


    # create none link
    start = len(a.relations)+1
    for i, pair in enumerate(pairs):
        id = start+i
        relation = Relation("R"+str(id),"None",argnames=['Arg1', 'Arg2'],references=pair)
        a.relations[relation.id] = relation
    
    badTerms = a.getBadTerms()
    if len(badTerms) == 0:
        text = open(TEXT_FILE).read()
        out_text = open(TEXT_FILE_OUTPUT,"w")
        out_text.write(text)
        a.write(ANNOTATION_FILE_OUTPUT)

    else: 
        spans = []
        for t in badTerms:
            spans += [[t.spans[0][0], t.spans[-1][1]]]

        text = open(TEXT_FILE).read()
        spans = mergeSpans(spans)
        spans = getFullSentences(text,spans)
        spans = mergeSpans(spans)
        #they are sorted after merging

        out_text = open(TEXT_FILE_OUTPUT,"w")

        text = removeSpans(text,spans)
        out_text.write(text)

        #removed spans
        a.deleteSpans(spans)
        a.write(ANNOTATION_FILE_OUTPUT)

    return ANNOTATION_FILE.split("/")[-1]

def masked_txt():
    """Masked Sensitive Informtion"""
    pass

def clean_ann(ANNOTATION_FILE, ANNOTATION_FILE_OUTPUT):
    
    # read file
    lines = []
    with open(ANNOTATION_FILE, 'r') as fp:
        lines = fp.readlines()

    # write file
    with open(ANNOTATION_FILE_OUTPUT, 'w') as fp:
        for line in lines:
            if line[0] != "#":
                fp.write(line)

def main():

    # path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/gold-ADR/German/lifeline/test"
    # target_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/full-ADR/German/lifeline/test"
    path = sys.argv[1]
    target_path = sys.argv[2]
    
    for file in os.listdir(path):
        if file.endswith(".ann"):
            name = file.split(".")[0]
            # import ipdb; ipdb.set_trace()
            ann_file = os.path.join(path, file)
            txt_file = os.path.join(path, name+'.txt')
            complex_anno = addNoneAnnotations(ann_file,txt_file, os.path.join(target_path, name+'.ann'),os.path.join(target_path, name+'.txt'))

    # path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/Keepha-ADR/Japanese/ja_forum"
    # tmp_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/Keepha-ADR/Japanese/ja_forum_rms"
    # target_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/processed-ADR/Japanese/ja_forum"
    # gold_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/gold-ADR/Japanese/ja_forum"

    # for dir in [tmp_path, target_path]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
                
    # # Re-create & detect files
    # c_anns = []
    # for file in os.listdir(path):
    #     if file.endswith(".ann"):
    #         name = file.split(".")[0]
    #         # import ipdb; ipdb.set_trace()
    #         ann_file = os.path.join(path, file)
    #         txt_file = os.path.join(path, name+'.txt')
    #         complex_anno = addNoneAnnotations(ann_file,txt_file, os.path.join(target_path, name+'.ann'),os.path.join(target_path, name+'.txt'))
    #         if complex_anno:
    #             c_anns.append(complex_anno)
                

    # # Distribute samples across train-dev-test
    # dirs = os.listdir(tmp_path)
    # anns = [ fname for fname in dirs if fname.endswith('.ann')]
    # non_c_anns = list(set(anns) - set(c_anns))

    # test = random.sample(non_c_anns, int(len(non_c_anns)*.2))
    # non_c_anns = list(set(non_c_anns) - set(test))
    # dev = random.sample(non_c_anns, int(len(non_c_anns)*.16))
    # non_c_anns = list(set(non_c_anns) - set(dev))
    # train = non_c_anns

    # test += random.sample(c_anns, int(len(c_anns)*.2))
    # c_anns = list(set(c_anns) - set(test))
    # dev += random.sample(c_anns, int(len(c_anns)*.16))
    # c_anns = list(set(c_anns) - set(dev))
    # train += c_anns

    # train_dir = os.path.join(target_path, "train")
    # dev_dir = os.path.join(target_path, "dev")
    # test_dir = os.path.join(target_path, "test")
    # for dir in [train_dir, dev_dir, test_dir]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    
    # for file in train:
    #     if not os.path.exists(os.path.join(train_dir, file)):
    #         shutil.copy(os.path.join(tmp_path, file), os.path.join(train_dir, file))
    #         shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(train_dir, file.split(".")[0]+".txt"))
    # for file in dev:
    #     if not os.path.exists(os.path.join(dev_dir, file)):
    #         shutil.copy(os.path.join(tmp_path, file), os.path.join(dev_dir, file))
    #         shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(dev_dir, file.split(".")[0]+".txt"))
    # for file in test:
    #     if not os.path.exists(os.path.join(test_dir, file)):
    #         shutil.copy(os.path.join(tmp_path, file), os.path.join(test_dir, file))
    #         shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(test_dir, file.split(".")[0]+".txt"))
    
    # print('Processed Data Created.')

    # # gold_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/gold-ADR/Japanese/ja_forum"
    # # target_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/processed-ADR/Japanese/ja_forum"
    # train_dir_ = os.path.join(target_path, "train")
    # dev_dir_ = os.path.join(target_path, "dev")
    # test_dir_ = os.path.join(target_path, "test")
    # for dir in [train_dir_, dev_dir_, test_dir_]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)

    # for dir in [gold_path, os.path.join(gold_path, 'train'), os.path.join(gold_path, 'dev'), os.path.join(gold_path, 'test')]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)

    # for file in os.listdir(train_dir):
    #     shutil.copy(os.path.join(path, file), os.path.join(gold_path, 'train', file))
        
    # for file in os.listdir(dev_dir):
    #     shutil.copy(os.path.join(path, file), os.path.join(gold_path, 'dev', file))
        
    # for file in os.listdir(test_dir):
    #     shutil.copy(os.path.join(path, file), os.path.join(gold_path, 'test', file))
    
    # print('Gold Data Created.')




if __name__ == "__main__":
    main()


