import os, sys
import random
import re
from collections import Counter
import shutil
import json


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
    for s in spans:
        id0 = text[:s[0]].rfind(".")
        id1 = text[s[1]:].find(".")
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

    path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/Keepha-ADR/German/lifeline"
    tmp_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/Keepha-ADR/German/lifeline_rms"
    target_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/processed-ADR/German/lifeline"

    for dir in [tmp_path, target_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # Re-create & detect files
    c_anns = []
    for file in os.listdir(path):
        if file.endswith(".ann"):
            name = file.split(".")[0]
            # import ipdb; ipdb.set_trace()
            ann_file = os.path.join(path, file)
            txt_file = os.path.join(path, name+'.txt')
            complex_anno = fixAnnotations(ann_file,txt_file, os.path.join(tmp_path, name+'.ann'),os.path.join(tmp_path, name+'.txt'))
            if complex_anno:
                c_anns.append(complex_anno)

    # import ipdb; ipdb.set_trace()
    # check c_anns: 291_lifeline_v2_7306_1_1652283642.ann, 350_lifeline_v2_9816_1_1651242282.ann, 79_lifeline_v2_9701_1_1651239622.ann
    # check non_c_anns: 329_lifeline_v2_7151_1_1648460358.ann, 59_lifeline_v2_7252_1_1649674890.ann

    # Distribute samples across train-dev-test
    dirs = os.listdir(tmp_path)
    anns = [ fname for fname in dirs if fname.endswith('.ann')]
    non_c_anns = list(set(anns) - set(c_anns))

    test = random.sample(non_c_anns, int(len(non_c_anns)*.2))
    non_c_anns = list(set(non_c_anns) - set(test))
    dev = random.sample(non_c_anns, int(len(non_c_anns)*.16))
    non_c_anns = list(set(non_c_anns) - set(dev))
    train = non_c_anns

    test += random.sample(c_anns, int(len(c_anns)*.2))
    c_anns = list(set(c_anns) - set(test))
    dev += random.sample(c_anns, int(len(c_anns)*.16))
    c_anns = list(set(c_anns) - set(dev))
    train += c_anns

    train_dir = os.path.join(target_path, "train")
    dev_dir = os.path.join(target_path, "dev")
    test_dir = os.path.join(target_path, "test")
    for dir in [train_dir, dev_dir, test_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for file in train:
        if not os.path.exists(os.path.join(train_dir, file)):
            shutil.copy(os.path.join(tmp_path, file), os.path.join(train_dir, file))
            shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(train_dir, file.split(".")[0]+".txt"))
    for file in dev:
        if not os.path.exists(os.path.join(dev_dir, file)):
            shutil.copy(os.path.join(tmp_path, file), os.path.join(dev_dir, file))
            shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(dev_dir, file.split(".")[0]+".txt"))
    for file in test:
        if not os.path.exists(os.path.join(test_dir, file)):
            shutil.copy(os.path.join(tmp_path, file), os.path.join(test_dir, file))
            shutil.copy(os.path.join(tmp_path, file.split(".")[0]+".txt"), os.path.join(test_dir, file.split(".")[0]+".txt"))


if __name__ == "__main__":
    main()


   # ann_file = os.path.join(path, "279_lifeline_v2_7465_1_1648460698.ann")
    # txt_file = os.path.join(path, "279_lifeline_v2_7465_1_1648460698.txt")
    # name = "279_lifeline_v2_7465_1_1648460698"
    # fixAnnotations(ann_file,txt_file, os.path.join(target_path, name+'.ann'),os.path.join(target_path, name+'.txt'))

    # all_c_anns = {"lifeline_de_curated": ['213_lifeline_v2_9773_1_1651242121.ann', '525_lifeline_v2_2938_1_1643294997.ann', '196_lifeline_v2_4258_1_1649668753.ann', '817_lifeline_v2_8546_1_1649244380.ann', '817_lifeline_v2_8546_1_1649244380.ann', '253_lifeline_v2_1711_1_1644400231.ann', '253_lifeline_v2_1711_1_1644400231.ann', '246_lifeline_v2_9739_1_1651241827.ann', '314_lifeline_v2_7358_1_1652456368.ann', '574_lifeline_v2_8580_1_1649244512.ann', '574_lifeline_v2_8580_1_1649244512.ann', '323_lifeline_v2_7284_1_1646214111.ann', '637_lifeline_v2_9081_1_1649674443.ann', '252_lifeline_v2_7501_1_1648460924.ann', '291_lifeline_v2_7306_1_1652283642.ann', '309_lifeline_v2_526_1_1649670386.ann', '309_lifeline_v2_526_1_1649670386.ann', '199_lifeline_v2_8504_1_1649244001.ann', '302_lifeline_v2_1381_1_1643102630.ann', '62_lifeline_v2_8363_1_1649243894.ann', '288_lifeline_v2_8722_1_1652452207.ann', '57_lifeline_v2_9271_1_1652369751.ann', '80_lifeline_v2_9639_1_1651239558.ann', '330_lifeline_v2_1287_1_1647256622.ann', '203_lifeline_v2_8035_1_1652349495.ann', '68_lifeline_v2_151_1_1643299726.ann', '68_lifeline_v2_151_1_1643299726.ann', '83_lifeline_v2_9617_1_1651239486.ann', '79_lifeline_v2_9701_1_1651239622.ann', '285_lifeline_v2_5610_1_1648458781.ann', '396_lifeline_v2_3177_1_1647859322.ann', '192_lifeline_v2_475_1_1647256396.ann', '414_lifeline_v2_1056_1_1644398664.ann', '414_lifeline_v2_1056_1_1644398664.ann', '350_lifeline_v2_9816_1_1651242282.ann', '200_lifeline_v2_5183_1_1643297943.ann', '477_lifeline_v2_7771_1_1648461633.ann', '349_lifeline_v2_9980_1_1651242704.ann', '277_lifeline_v2_478_1_1643621648.ann', '487_lifeline_v2_2624_1_1647857881.ann', '69_lifeline_v2_9303_1_1652457034.ann', '269_lifeline_v2_7976_1_1648465273.ann', '75_lifeline_v2_9732_1_1651241753.ann', '436_lifeline_v2_2590_1_1649246186.ann', '436_lifeline_v2_2590_1_1649246186.ann', '396_lifeline_v2_253_1_1649674488.ann', '445_lifeline_v2_2902_1_1643294953.ann', '205_lifeline_v2_7344_1_1648460631.ann', '228_lifeline_v2_5429_1_1652282183.ann', '356_lifeline_v2_6288_1_1643299480.ann', '231_lifeline_v2_177_1_1647857188.ann'],
    #                         "lexapro_ade-202211r": ['twjp_200-220.ann', 'twjp_540-560.ann', 'twjp_320-340.ann', 'twjp_060-080.ann', 'twjp_120-140.ann']
    #                         }

    # c_anns = all_c_anns[dataset]
    # # c_anns = ['77_lifeline_v2_9981_0_1649151794.ann', '83_lifeline_v2_9972_0_1649078290.ann', '76_lifeline_v2_9765_0_1648801104.ann', '80_lifeline_v2_6675_1_1648459681.ann', '525_lifeline_v2_2938_1_1643294997.ann', '376_lifeline_v2_8686_1_1649671738.ann', '78_lifeline_v2_2600_0_1640253541.ann', '83_lifeline_v2_7643_0_1648203487.ann', '66_lifeline_v2_8326_0_1646923263.ann', '817_lifeline_v2_8546_1_1649244380.ann', '817_lifeline_v2_8546_1_1649244380.ann', '68_lifeline_v2_6967_0_1647609312.ann', '63_lifeline_v2_9235_0_1652456968.ann', '57_lifeline_v2_9444_0_1648799657.ann', '574_lifeline_v2_8580_1_1649244512.ann', '574_lifeline_v2_8580_1_1649244512.ann', '64_lifeline_v2_4145_0_1647942696.ann', '279_lifeline_v2_7465_1_1648460698.ann', '69_lifeline_v2_4334_0_1641291088.ann', '59_lifeline_v2_9325_0_1648715455.ann', '637_lifeline_v2_9081_1_1649674443.ann', '77_lifeline_v2_9589_0_1648799891.ann', '63_lifeline_v2_8071_0_1646922610.ann', '71_lifeline_v2_8977_0_1642409457.ann', '83_lifeline_v2_7849_0_1648204432.ann', '305_lifeline_v2_8981_1_1652434359.ann', '305_lifeline_v2_8981_1_1652434359.ann', '80_lifeline_v2_9133_0_1648714935.ann', '309_lifeline_v2_526_1_1649670386.ann', '199_lifeline_v2_8504_1_1649244001.ann', '199_lifeline_v2_8504_1_1649244001.ann', '80_lifeline_v2_3081_0_1640434995.ann', '78_lifeline_v2_3803_0_1640781526.ann', '83_lifeline_v2_9890_0_1649077551.ann', '76_lifeline_v2_7467_0_1638357744.ann', '83_lifeline_v2_7631_0_1646665646.ann', '493_lifeline_v2_7629_1_1648461452.ann', '493_lifeline_v2_7629_1_1648461452.ann', '70_lifeline_v2_6893_0_1646126152.ann', '63_lifeline_v2_5649_0_1641644417.ann', '73_lifeline_v2_3223_0_1645090011.ann', '83_lifeline_v2_7842_0_1648204430.ann', '70_lifeline_v2_7787_0_1648632950.ann', '57_lifeline_v2_9271_1_1652369751.ann', '77_lifeline_v2_7239_1_1649671101.ann', '328_lifeline_v2_6220_1_1649674844.ann', '72_lifeline_v2_4020_0_1641222425.ann', '74_lifeline_v2_5417_0_1647943744.ann', '83_lifeline_v2_9617_1_1651239486.ann', '83_lifeline_v2_9617_1_1651239486.ann', '73_lifeline_v2_6478_0_1647599934.ann', '77_lifeline_v2_6427_0_1642413439.ann', '70_lifeline_v2_5224_0_1645786319.ann', '186_lifeline_v2_8659_1_1649244869.ann', '207_lifeline_v2_7106_1_1648460271.ann', '396_lifeline_v2_3177_1_1647859322.ann', '192_lifeline_v2_475_1_1647256396.ann', '414_lifeline_v2_1056_1_1644398664.ann', '414_lifeline_v2_1056_1_1644398664.ann', '57_lifeline_v2_8478_0_1647087863.ann', '350_lifeline_v2_9816_1_1651242282.ann', '477_lifeline_v2_7771_1_1648461633.ann', '349_lifeline_v2_9980_1_1651242704.ann', '82_lifeline_v2_8887_0_1647160968.ann', '72_lifeline_v2_6550_0_1646047648.ann', '436_lifeline_v2_2590_1_1649246186.ann', '436_lifeline_v2_2590_1_1649246186.ann', '79_lifeline_v2_9621_0_1648799967.ann', '70_lifeline_v2_8147_0_1646922650.ann', '83_lifeline_v2_8777_0_1648633803.ann', '356_lifeline_v2_6288_1_1643299480.ann']
    # # there are unique 63 unique ann files
    # # target_path = "/mnt/beegfs/home/huisyuan/keepha/m-ADR/data/Keepha-ADR/Japanese/tweets_rms/"
    # if not os.path.exists(target_path):
    #     os.makedirs(target_path)

    # # # Check the num of unique files
    # # print(len(c_anns))
    # # print(len(list(set(c_anns))))
    # # print([i for i in c_anns if c_anns.count(i)>1])
    
    # # # Re-create Japanese-Overlapping
    # if language=="ja":
    #     for file in os.listdir(path):
    #         if file.endswith('.ann'):
    #             clean_ann(os.path.join(path,file), os.path.join(tmp_path,file))

    # # # Re-create the complicated ann/txt
    # for ann_file in c_anns:
    #     name = ann_file.split(".")[0]
    #     txt_file = os.path.join(path, name+'.txt')
    #     if language=="de":
    #         ann_file = os.path.join(path, ann_file)
    #     if language=="ja":
    #         ann_file = os.path.join(tmp_path, ann_file)
       
    #     fixAnnotations(ann_file,txt_file, os.path.join(target_path, name+'.ann'),os.path.join(target_path, name+'.txt'))
    

    # targe_path = "./"
    # # Re-create the complicated ann/txt
    # fixAnnotations(ann_file,txt_file, os.path.join(target_path, name+'.ann'),os.path.join(target_path, name+'.txt'))

    