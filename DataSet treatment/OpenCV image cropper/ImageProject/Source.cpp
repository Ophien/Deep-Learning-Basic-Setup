// printf, scan, fprintf, fscanf
#include <stdio.h>

// string
#include <string.h>

// vector
#include <vector>
#include <map>

// directory
#include <filesystem>

// windows 
#include <windows.h>
#include <tchar.h> 
#include <strsafe.h>
#pragma comment(lib, "User32.lib")

// opencv core
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Point getUpperLeftPoint(Mat &im) {
	Vec3b row[5];
	Vec3b column[5];

	int xStart = 0;
	int yStart = 0;

	bool end = false;
	for (int x = 0; x < im.cols - 5 && !end; x++) {
		for (int y = 0; y < im.rows - 5 && !end; y++) {
			row[0] = im.at<cv::Vec3b>(y, x);
			row[1] = im.at<cv::Vec3b>(y, x + 1);
			row[2] = im.at<cv::Vec3b>(y, x + 2);
			row[3] = im.at<cv::Vec3b>(y, x + 3);
			row[4] = im.at<cv::Vec3b>(y, x + 4);

			column[0] = im.at<cv::Vec3b>(y, x);
			column[1] = im.at<cv::Vec3b>(y + 1, x);
			column[2] = im.at<cv::Vec3b>(y + 2, x);
			column[3] = im.at<cv::Vec3b>(y + 3, x);
			column[4] = im.at<cv::Vec3b>(y + 4, x);

			float rowr = 0;
			float rowg = 0;
			float rowb = 0;
			for (int i = 0; i < 5; i++) {
				uchar red = row[i][2];
				uchar green = row[i][1];
				uchar blue = row[i][0];

				rowr += (float)red;
				rowg += (float)green;
				rowb += (float)blue;

				red = column[i][2];
				green = column[i][1];
				blue = column[i][0];

				rowr += (float)red;
				rowg += (float)green;
				rowb += (float)blue;
			}
			rowr /= 10.0f;
			rowg /= 10.0f;
			rowb /= 10.0f;

			float dist =
				(rowr - 255.0f) * (rowr - 255.0f) +
				(rowg - 0.0f) * (rowg - 0.0f) +
				(rowb - 0.0f) * (rowb - 0.0f);

			dist = sqrt(dist);

			if (dist < 2)
			{
				xStart = x;
				yStart = y;
				//end = true;
			}
		}
	}


	xStart++;
	yStart++;

	Point ret(xStart, yStart);
	return ret;
}

Point getLowerRightPoint(Mat &im) {
	Vec3b row[5];
	Vec3b column[5];

	int xStart = 0;
	int yStart = 0;

	bool end = false;
	for (int x = im.cols - 1; x >= 4 && !end; x--) {
		for (int y = im.rows - 1; y >= 4 && !end; y--) {
			row[0] = im.at<cv::Vec3b>(y, x);
			row[1] = im.at<cv::Vec3b>(y, x - 1);
			row[2] = im.at<cv::Vec3b>(y, x - 2);
			row[3] = im.at<cv::Vec3b>(y, x - 3);
			row[4] = im.at<cv::Vec3b>(y, x - 4);

			column[0] = im.at<cv::Vec3b>(y, x);
			column[1] = im.at<cv::Vec3b>(y - 1, x);
			column[2] = im.at<cv::Vec3b>(y - 2, x);
			column[3] = im.at<cv::Vec3b>(y - 3, x);
			column[4] = im.at<cv::Vec3b>(y - 4, x);

			float rowr = 0;
			float rowg = 0;
			float rowb = 0;
			for (int i = 0; i < 5; i++) {
				uchar red = row[i][2];
				uchar green = row[i][1];
				uchar blue = row[i][0];

				rowr += (float)red;
				rowg += (float)green;
				rowb += (float)blue;

				red = column[i][2];
				green = column[i][1];
				blue = column[i][0];

				rowr += (float)red;
				rowg += (float)green;
				rowb += (float)blue;
			}
			rowr /= 10.0f;
			rowg /= 10.0f;
			rowb /= 10.0f;

			float dist =
				(rowr - 255.0f) * (rowr - 255.0f) +
				(rowg - 0.0f) * (rowg - 0.0f) +
				(rowb - 0.0f) * (rowb - 0.0f);

			dist = sqrt(dist);

			if (dist < 2)
			{
				xStart = x;
				yStart = y;
				//end = true;
			}
		}
	}

	xStart--;
	yStart--;

	Point ret(xStart, yStart);
	return ret;
}

std::wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

std::string LPCSTR2String(wchar_t* txt) {
	wstring ws(txt);
	// your new String
	string str(ws.begin(), ws.end());

	return str;
}

vector<string> readAllImages(string path) {
	vector<string> files;

	HANDLE hFind;
	WIN32_FIND_DATA data;
	
	std::wstring stemp = s2ws(path);
	LPCWSTR result = stemp.c_str();
	
	hFind = FindFirstFile(result, &data);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			string file = LPCSTR2String(data.cFileName);
			files.push_back(file);
		} while (FindNextFile(hFind, &data));
		FindClose(hFind);
	}

	return files;
}

const vector<string> explode(const string& s, const char& c)
{
	string buff{ "" };
	vector<string> v;

	for (auto n : s)
	{
		if (n != c) buff += n; else
			if (n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if (buff != "") v.push_back(buff);

	return v;
}

void cropAllImages(string readingPath, string savingPath, string readingExtension, string savingExtension) {
	vector<string> files = readAllImages(readingPath + readingExtension);

	const int nnSize = 227;
	for (int i = 0; i < files.size(); i++) {
		string file = files.at(i);
		string img2read = readingPath + file;
		vector<string> fileTokenized{ explode(file, '.') };

		string img2save = savingPath + "CROPPED_" + fileTokenized[0] + savingExtension;

		// read image
		Mat im = imread(img2read, 1);

		Point upperLeft = getUpperLeftPoint(im);
		Point lowerRight = getLowerRightPoint(im);

		if (lowerRight.x == -1 && lowerRight.y == -1 ||
			upperLeft.x == 0 && upperLeft.y == 0 &&
			lowerRight.x == -1 && lowerRight.y == -1 ||
			upperLeft.x == lowerRight.x ||
			upperLeft.y == lowerRight.y)
			continue;

		cv::Rect rect(upperLeft, lowerRight);
		cv::Mat miniMat;
		miniMat = im(rect);

		Mat resized;
		cv::resize(miniMat, resized, cv::Size(nnSize, nnSize));

		imwrite(img2save, resized);
		printf("Images proc.: %i of %i\n", i + 1, files.size());
	}
	//namedWindow("Colored", WINDOW_AUTOSIZE);
	//imshow("Colored", miniMat);
	//waitKey(0);
}

void createImageClassFile(string readingPath, string savingPath, string readingExtension) {
	vector<string> files = readAllImages(readingPath + readingExtension);
	map<string, int> classes;
	map<string, int> classesCounter;
	for (int i = 0; i < files.size(); i++) {
		string file = files.at(i);
		vector<string> splited = explode(file, '_');
		string img_class = splited.at(splited.size() - 1);
		if (classes.find(img_class) == classes.end()) {
			classes[img_class] = 1;
			classesCounter[img_class] = 1;
		}
	}

	int classCounter = 0;
	map<string, int>::iterator it = classes.begin();
	for (;it != classes.end();it++) {
		classes[(*it).first] = classCounter;
		classCounter++;
	}

	FILE* file;
	file = fopen((savingPath + "training.txt").c_str(), "w");
	for (int i = 0; i < files.size(); i++) {
		string fileName = files.at(i);
		vector<string> splited = explode(fileName, '_');
		string img_class = splited.at(splited.size() - 1);
		int img_id = classes[img_class];
		fprintf(file, "%s %i\n", fileName.c_str(), img_id);

		classesCounter[img_class] = classesCounter[img_class] + 1;
	}
	fclose(file);

	FILE* classesStats;
	classesStats = fopen((savingPath + "stats.txt").c_str(), "w");
	it = classesCounter.begin();
	for (;it != classesCounter.end();it++) {
		string className = (*it).first;
		vector<string> explodedName = explode(className, '.');
		int id = (*it).second;
		fprintf(classesStats, "%s %i\n", explodedName.at(0).c_str(), id);
	}
	fclose(classesStats);

	FILE* classesIds;
	classesIds = fopen((savingPath + "IDs.txt").c_str(), "w");
	it = classes.begin();
	for (;it != classes.end();it++) {
		string className = (*it).first;
		vector<string> explodedName = explode(className, '.');
		int id = (*it).second;
		fprintf(classesIds, "%s %i\n", explodedName.at(0).c_str(), id);
	}
	fclose(classesIds);
}

int main(int argc, char* argv[]) {
	createImageClassFile("F:/CLOTHS_CROPPED/", "F:/CLOTHS_CROPPED/", "*.png");
}