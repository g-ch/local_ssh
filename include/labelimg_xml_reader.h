//
// Created by cc on 2020/3/16.
//

#ifndef LOCAL_SSH_LABELIMG_XML_READER_H
#define LOCAL_SSH_LABELIMG_XML_READER_H

#include "tinyxml2.h"
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cerrno>

using namespace tinyxml2;

typedef struct Object{
    std::string label;
    int x_min;
    int y_min;
    int x_max;
    int y_max;
}Object;

XMLElement* nextXElement(XMLElement *element, int x){
    for(int i=0; i<x; i++)
    {
        element = element->NextSiblingElement();
    }
    return element;
}

bool readLabelIMGObjectDetectionXML(std::string xml_path, std::string &img_path, int &img_width, int &img_height, int &img_depth, std::vector<Object> &objects)
{
    /// Input label_img object detection label XML file path and output path, width, height, depth, and objects information in the XML
    XMLDocument file;
    const char *xml_path_cstr = xml_path.c_str();
    if(file.LoadFile(xml_path_cstr)){ // 0 means success. Error IDs > 0
        std::cout << "Failed to read file " << xml_path << std::endl;
        std::cout << "Error ID " << file.ErrorID() << std::endl;
        return false;
    }else{
        XMLElement *root = file.RootElement();
        XMLElement *child_element = root->FirstChildElement();

        child_element = child_element->NextSiblingElement();
        child_element = child_element->NextSiblingElement();
        img_path = child_element->FirstChild()->Value();

        child_element = child_element->NextSiblingElement();
        child_element = child_element->NextSiblingElement();
        XMLElement *img_size_element = child_element->FirstChildElement();

        img_width = atoi(img_size_element->FirstChild()->Value());
        img_size_element = img_size_element->NextSiblingElement();
        img_height = atoi(img_size_element->FirstChild()->Value());
        img_size_element = img_size_element->NextSiblingElement();
        img_depth = atoi(img_size_element->FirstChild()->Value());

        child_element = nextXElement(child_element, 2);

        while(child_element){
            Object object_this;
            XMLElement *object_element = child_element->FirstChildElement();
            object_this.label = object_element->FirstChild()->Value();
//            std::cout <<"label "<<object_this.label<<std::endl;

            object_element = nextXElement(object_element, 4);
            XMLElement *size_element = object_element->FirstChildElement();
            object_this.x_min = atoi(size_element->FirstChild()->Value());
            size_element = size_element->NextSiblingElement();
            object_this.y_min = atoi(size_element->FirstChild()->Value());
            size_element = size_element->NextSiblingElement();
            object_this.x_max = atoi(size_element->FirstChild()->Value());
            size_element = size_element->NextSiblingElement();
            object_this.y_max = atoi(size_element->FirstChild()->Value());

            objects.push_back(object_this);
            child_element = child_element->NextSiblingElement();
        }

        return true;
    }
}



#endif //LOCAL_SSH_LABELIMG_XML_READER_H
