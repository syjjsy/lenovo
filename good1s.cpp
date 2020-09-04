#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <conio.h>
#include <string>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")
#include <pylon/PylonIncludes.h>
//#include <pylon/usb/BaslerUsbCamera.h>
#include <pylon/PylonIncludes.h>
#include <pylon/gige/BaslerGigECamera.h>
#include <pylon/gige/BaslerGigEInstantCamera.h>
#include <pylon/gige/_BaslerGigECameraParams.h>
#include <cstdlib>
#include <vector>
#include <numeric>
#include<windows.h>
#include <time.h> 
#include <mutex>
#include <thread>
#include <pthread.h>
#include <condition_variable>
//typedef Pylon::CBaslerUsbCamera Camera_u;
typedef Pylon::CBaslerGigECamera Camera_t;
//typedef Pylon::CBaslerGigEInstantCamera Camera_t;
#ifdef PYLON_WIN_BUILD
#include <pylon/PylonGUI.h>    
#endif
//using namespace Basler_UsbCameraParams;
//using namespace Basler_UsbStreamParams;
using namespace Basler_GigECameraParams;
using namespace Basler_GigEStreamParams;

using namespace std;
using namespace cv;
using namespace Pylon;
using namespace GenApi;

int receive_data_flag = 0;
mutex data_mutex;
condition_variable data_var;


string savepath = "D://APP//pylonpic//";
string savepath0;
string savepath_1;
string savepath_2;

int height = 1200;
int width = 1600;
int OffsetX = 0;
int OffsetY = 0;
int Exp = 150000;

Mat src = Mat::zeros(height, width, CV_8UC1);
Mat src1 = Mat::zeros(height, width, CV_8UC1);


class CHeartbeatHelper
{
public:
	explicit CHeartbeatHelper(CBaslerGigECamera& Camera)
		: m_pHeartbeatTimeout(NULL)
	{
		// m_pHeartbeatTimeout may be NULL
		//m_pHeartbeatTimeout = camera.GetTLNodeMap().GetNode("HeartbeatTimeout");
		m_pHeartbeatTimeout = Camera.GetTLNodeMap()->GetNode("HeartbeatTimeout");
	}

	bool SetValue(int64_t NewValue)
	{
		// Do nothing if no heartbeat feature is available.
		if (!m_pHeartbeatTimeout.IsValid())
			return false;

		// Apply the increment and cut off invalid values if neccessary.
		int64_t correctedValue = NewValue - (NewValue % m_pHeartbeatTimeout->GetInc());

		m_pHeartbeatTimeout->SetValue(correctedValue);
		return true;
	}

	bool SetMax()
	{
		// Do nothing if no heartbeat feature is available.
		if (!m_pHeartbeatTimeout.IsValid())
			return false;

		int64_t maxVal = m_pHeartbeatTimeout->GetMax();
		return SetValue(maxVal);
	}

protected:
	GenApi::CIntegerPtr m_pHeartbeatTimeout; // Pointer to the node, will be NULL if no node exists.
};



int newfold(string folderPath)
{
	bool flag = CreateDirectory(folderPath.c_str(), NULL);
	if (flag == true) { cout << "文件夹" << folderPath << "创建成功" << endl; }
	else { cout << "Directory " << folderPath << " already exists." << endl; }
	return 0;
}

void* camera1(void* args)
{
	Pylon::PylonAutoInitTerm autoInitTerm;
	try
	{

		CTlFactory& TlFactory = CTlFactory::GetInstance();
		ITransportLayer* pTl = TlFactory.CreateTl(Camera_t::DeviceClass());


		if (!pTl)
		{
			cerr << "Failed to create transport layer!" << endl;
			//            return -1;
		}

		DeviceInfoList_t devices;
		if (0 == pTl->EnumerateDevices(devices))
		{
			cerr << "No camera present!" << endl;
			//            return 2;
		}

		Camera_t Camera(pTl->CreateDevice(devices[0]));

		CHeartbeatHelper headbeatHelper(Camera);
		headbeatHelper.SetValue(1000);

		Camera.Open();

		Camera_t::StreamGrabber_t StreamGrabber(Camera.GetStreamGrabber(0));

		StreamGrabber.Open();

		Camera.PixelFormat.SetValue(PixelFormat_Mono8);
		//Camera.PixelFormat.SetValue(PixelFormat_BayerRG8);

		Camera.OffsetX.SetValue(OffsetX);
		Camera.OffsetY.SetValue(OffsetY);
		Camera.Width.SetValue(width);
		Camera.Height.SetValue(height);
		GenApi::IEnumEntry* frameStart = Camera.TriggerSelector.GetEntry(TriggerSelector_FrameStart);

		Camera.AcquisitionMode.SetValue(AcquisitionMode_Continuous);

		Camera.ExposureMode.SetValue(ExposureMode_Timed);
		//Camera.ExposureTimeAbs.SetValue(20);
		Camera.ExposureTimeRaw.SetValue(Exp);
		Camera.AcquisitionFrameRateEnable.SetValue(TRUE);
		Camera.AcquisitionFrameRateAbs.SetValue(1);

		Camera.GainAuto.SetValue(GainAuto_Off);

		const size_t ImageSize = (size_t)(Camera.PayloadSize.GetValue());
		ushort* const pBuffer = new ushort[ImageSize];

		StreamGrabber.MaxBufferSize.SetValue(ImageSize);

		StreamGrabber.MaxNumBuffer.SetValue(1);

		StreamGrabber.PrepareGrab();

		const StreamBufferHandle hBuffer =
			StreamGrabber.RegisterBuffer(pBuffer, ImageSize);

		StreamGrabber.QueueBuffer(hBuffer, NULL);
		if (frameStart && GenApi::IsAvailable(frameStart))
		{
			Camera.TriggerSelector.SetValue(TriggerSelector_FrameStart);
			Camera.TriggerMode.SetValue(TriggerModeEnums(0));
		}
		cout << "Camera1 open!" << endl;

		GrabResult Result;

		Camera.AcquisitionStart.Execute();

		int num = 0;
		while (1)
		{
			std::unique_lock<std::mutex>lck1(data_mutex);
			data_var.wait(lck1, [] { return !receive_data_flag; });
			printf("请您按下回车采集@_@");
			while (cin.get() == '\n')
			{
				break;
			}

			//if (StreamGrabber.GetWaitObject().Wait(30000))
			//{
				//                                GrabResult Result;
				//Sleep(1000);
			StreamGrabber.RetrieveResult(Result);
			//if (Result.Succeeded())
			//{
			CGrabResultImageRef resultsrc = Result.GetImage();
			//cout << resultsrc.GetHeight() << " " << resultsrc.GetWidth() << endl;
			Mat src = cv::Mat(resultsrc.GetHeight(), resultsrc.GetWidth(), CV_8UC1, (uint8_t*)resultsrc.GetBuffer());

			//memcpy(src.data, Result.Buffer(), src.cols * src.rows * sizeof(uchar));


			imwrite(savepath_1 + to_string(num) + ".bmp", src);
			cout << 1 << endl;
			num++;
			//}
		//}
		    StreamGrabber.QueueBuffer(Result.Handle(), Result.Context());
			Sleep(1000);
			receive_data_flag = 1;
			//cout << receive_data_flag << " lalalalaalala" << endl;
			data_var.notify_one();
		}


		StreamGrabber.DeregisterBuffer(hBuffer);

		StreamGrabber.FinishGrab();

		StreamGrabber.Close();

		Camera.Close();
		delete[] pBuffer;


	}

	catch (const GenericException& e)
	{
		// Error handling
	//        cerr << "An exception occurred." << endl
	//            << e.GetDescription() << endl;
	}


	return 0;
}

void* camera2(void* args)
{

	Pylon::PylonAutoInitTerm autoInitTerm;
	try
	{
		CTlFactory& TlFactory = CTlFactory::GetInstance();
		ITransportLayer* pTl = TlFactory.CreateTl(Camera_t::DeviceClass());


		if (!pTl)
		{
			cerr << "Failed to create transport layer!" << endl;
			//            return -1;
		}

		DeviceInfoList_t devices;
		if (0 == pTl->EnumerateDevices(devices))
		{
			cerr << "No camera present!" << endl;
			//            return 2;
		}

		Camera_t Camera(pTl->CreateDevice(devices[1]));

		CHeartbeatHelper headbeatHelper(Camera);
		headbeatHelper.SetValue(500);

		Camera.Open();

		Camera_t::StreamGrabber_t StreamGrabber(Camera.GetStreamGrabber(0));

		StreamGrabber.Open();

		Camera.PixelFormat.SetValue(PixelFormat_Mono8);
		//Camera.PixelFormat.SetValue(PixelFormat_BayerRG8);

		Camera.OffsetX.SetValue(OffsetX);
		Camera.OffsetY.SetValue(OffsetY);
		Camera.Width.SetValue(width);
		Camera.Height.SetValue(height);
		GenApi::IEnumEntry* frameStart = Camera.TriggerSelector.GetEntry(TriggerSelector_FrameStart);

		Camera.AcquisitionMode.SetValue(AcquisitionMode_Continuous);

		Camera.ExposureMode.SetValue(ExposureMode_Timed);
		//Camera.ExposureTimeAbs.SetValue(20);
		Camera.ExposureTimeRaw.SetValue(Exp);
		Camera.AcquisitionFrameRateEnable.SetValue(TRUE);
		Camera.AcquisitionFrameRateAbs.SetValue(1);

		Camera.GainAuto.SetValue(GainAuto_Off);

		const size_t ImageSize = (size_t)(Camera.PayloadSize.GetValue());
		ushort* const pBuffer = new ushort[ImageSize];

		StreamGrabber.MaxBufferSize.SetValue(ImageSize);

		StreamGrabber.MaxNumBuffer.SetValue(1);

		StreamGrabber.PrepareGrab();

		const StreamBufferHandle hBuffer =
			StreamGrabber.RegisterBuffer(pBuffer, ImageSize);

		StreamGrabber.QueueBuffer(hBuffer, NULL);
		if (frameStart && GenApi::IsAvailable(frameStart))
		{
			Camera.TriggerSelector.SetValue(TriggerSelector_FrameStart);
			Camera.TriggerMode.SetValue(TriggerModeEnums(0));
		}
		cout << "Camera2 open!" << endl;

		GrabResult Result;

		Camera.AcquisitionStart.Execute();

		int num1 = 0;
		while (1)
		{
			std::unique_lock<std::mutex>lck2(data_mutex);
			data_var.wait(lck2, [] {return receive_data_flag; });
			//if (StreamGrabber.GetWaitObject().Wait(30000))
			//{
			//Sleep(1000);
				//                                GrabResult Result;
			StreamGrabber.RetrieveResult(Result);
			//if (Result.Succeeded())
			//{
			CGrabResultImageRef resultsrc = Result.GetImage();
			//cout << resultsrc.GetHeight() << " " << resultsrc.GetWidth() << endl;
			Mat src1 = cv::Mat(resultsrc.GetHeight(), resultsrc.GetWidth(), CV_8UC1, (uint8_t*)resultsrc.GetBuffer());
			//memcpy(src1.data, Result.Buffer(), src1.cols * src1.rows * sizeof(uchar));

			imwrite(savepath_2 + to_string(num1) + ".bmp", src1);
			cout << 2 << endl;
			num1++;
			//}
		//}

		    StreamGrabber.QueueBuffer(Result.Handle(), Result.Context());
			Sleep(1000);
		//cout << "whywhywhywhy" << endl;
			receive_data_flag = 0;
			data_var.notify_one();
		}


		StreamGrabber.DeregisterBuffer(hBuffer);

		StreamGrabber.FinishGrab();

		StreamGrabber.Close();

		Camera.Close();
		delete[] pBuffer;


	}

	catch (const GenericException& e)
	{
		// Error handling
	//        cerr << "An exception occurred." << endl
	//            << e.GetDescription() << endl;
	}


	return 0;
}


int main(void* args)
{
	time_t t1 = time(0);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%H_%M_%S", localtime(&t1));
	string s = tmp;
	savepath0 = savepath + s + "//";
	newfold(savepath0);

	savepath_1 = savepath0 + to_string(1) + "//";
	newfold(savepath_1);
	savepath_2 = savepath0 + to_string(2) + "//";
	newfold(savepath_2);

	pthread_t threadA;
	pthread_t threadB;

	if (pthread_create(&threadA, NULL, &camera1, NULL) == -1)
	{
		printf("create camera thread error!\n");
		return 1;
	}
	if (pthread_create(&threadB, NULL, &camera2, NULL) == -1)
	{
		printf("create fuse thread error!\n");
		return 2;
	}
	while (1) {
	}


	if (pthread_join(threadA, NULL))
	{
		printf("camera thread error!\n");
		return -1;
	}
	if (pthread_join(threadB, NULL))
	{
		printf("fuse thread error!\n");
		return -2;
	}
}