import os
import pandas as pd
import cv2
import numpy as np
import glob
import xml.dom.minidom as minidom

def csv2xml(result,xmlpath,pic_shape):
    
    bigpic = pd.DataFrame(result.groupby(result['pic_name']).size())
    bigpic = bigpic.reset_index()
    len_bigpic = len(result.groupby(['pic_name']).size())
    
    for i in range(len_bigpic):
        bigpic_name = bigpic.loc[i,'pic_name']
        tmp = result[result['pic_name']==bigpic_name]
        objectnum = len(tmp)
        shape = pic_shape[bigpic_name]
        
        doc = minidom.Document()             
        annotations = doc.createElement('annotations')
        doc.appendChild(annotations)
        folder = doc.createElement('folder')
        annotations.appendChild(folder)
        folder.appendChild(doc.createTextNode('xml'))
        filename = doc.createElement('filename')
        annotations.appendChild(filename)
        filename.appendChild(doc.createTextNode(bigpic_name+'.jpg'))
        path = doc.createElement('path')
        annotations.appendChild(path)
        path.appendChild(doc.createTextNode(os.path.join(xmlpath,bigpic_name+'.jpg')))

        source = doc.createElement('source')
        annotations.appendChild(source)
        database = doc.createElement('database')
        source.appendChild(database)
        database.appendChild(doc.createTextNode('Unknown'))

        size = doc.createElement('size')
        annotations.appendChild(size)
        width = doc.createElement('width')
        size.appendChild(width)
        width.appendChild(doc.createTextNode(str(shape[1])))
        height = doc.createElement('height')
        size.appendChild(height)
        height.appendChild(doc.createTextNode(str(shape[0])))
        depth = doc.createElement('depth')
        size.appendChild(depth)
        depth.appendChild(doc.createTextNode('3'))
        segmented = doc.createElement('segmented')
        annotations.appendChild(segmented)
        segmented.appendChild(doc.createTextNode('0'))

        for j in range(objectnum):

            name1 = tmp.iloc[j,5]
            
            bigxmin = int(tmp.iloc[j,1])
            bigxmax = int(tmp.iloc[j,3])
            bigymin = int(tmp.iloc[j,2])
            bigymax = int(tmp.iloc[j,4])

            ###writexml
            object = doc.createElement('object')
            annotations.appendChild(object)

            name = doc.createElement('name')
            object.appendChild(name)
            name.appendChild(doc.createTextNode(name1))

            pose = doc.createElement('pose')
            object.appendChild(pose)
            pose.appendChild(doc.createTextNode('Unspecified'))

            truncated = doc.createElement('truncated')
            object.appendChild(truncated)
            truncated.appendChild(doc.createTextNode('0'))

            difficult = doc.createElement('difficult')
            object.appendChild(difficult)
            difficult.appendChild(doc.createTextNode('0'))

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin.appendChild(doc.createTextNode(str(bigxmin)))
            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin.appendChild(doc.createTextNode(str(bigymin)))
            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax.appendChild(doc.createTextNode(str(bigxmax)))
            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax.appendChild(doc.createTextNode(str(bigymax)))

        f = open(os.path.join(xmlpath, bigpic_name + '.xml'), 'w')
        doc.writexml(f)  
        f.close()
    