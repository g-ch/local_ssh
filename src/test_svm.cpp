#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

struct ppt
{
    float x;
    float y;
    char label;
};

int main()
{
    //训练一个简单的svm模型
    // step 1: 标记分类
    int labels[14] = { 'A','A','A','A','A','A','A','B','B','B','B','B','B','B' };
    //CvMat labelsMat = cvMat(14, 1, CV_32FC1, labels);
    Mat labelsMat(14, 1, CV_32SC1);
    for (int i = 0; i < labelsMat.rows; i++)
    {
        labelsMat.at<int>(i, 0) = labels[i];
    }

    int trainingData[14][2] = { { 110,204 },{ 105,306 },{ 102,410 },{ 99,511 },{ 93,610 },{ 89,713 },{ 89,817 },
                                { 173,208 },{ 175,313 },{ 167,415 },{ 163,514 },{ 160,612 },{ 156,716 },{ 152,819 } };
    Mat trainingDataMat(14, 2, CV_32FC1);
    for (int i = 0; i < trainingDataMat.rows; i++)
    {
        for (int j = 0; j < trainingDataMat.cols; j++)
        {
            trainingDataMat.at<float>(i, j) = trainingData[i][j];
        }
    }

    //step 2:设定训练参数
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    //迭代训练过程的中止条件，解决部分受约束二次最优问题。您可以指定的公差和/或最大迭代次数。
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, FLT_EPSILON));

    //step 3:训练
    Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
    svm->train(tData);
    svm->save("svmData.xml");

    // step 4: 利用训练好的模型进行预测
    ppt a;
    a.x = 163;
    a.y = 600;
    float data[2] = { a.x,a.y };
    Mat tmp(1, 2, CV_32FC1);
    for (int j = 0; j < tmp.cols; j++)
    {
        tmp.at<float>(0, j) = data[j];
    }

    a.label = (char)svm->predict(tmp);
    cout << a.label << endl;

    cv::Mat result;
    float rst = svm->predict(trainingDataMat, result);
    for (int i = 0; i < result.rows; i++){
        std::cout << (char)result.at<float>(i, 0)<<" ";
    }

}
