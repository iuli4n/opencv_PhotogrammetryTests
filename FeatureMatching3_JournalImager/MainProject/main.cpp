/******
JOURNAL PAGE EXTRACTOR FROM VIDEO OF FLIPPING PAGES

The way it's supopsed to work:
- detect a clean page and train a set of keypoints on it
- as the next frames go by, see if the next frames are of the same page or not (based on keypoints check)
- if detected new page, save the old one and train on the new one

***/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include "image_checkers.cpp"

using namespace cv;
using namespace std;

// =======================
// Helper Functions
// =======================

// Get current time in milliseconds
long getTimeMS() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


// Crop and convert page region to grayscale
Mat capturePageRegion(const Mat& frame, const Rect& area) {
    Mat cropped = frame(area);//.clone();
    Mat gray;
    cvtColor(cropped, gray, COLOR_RGB2GRAY);
    return gray;
}


// Detect features and compute descriptors
void detectFeatures(
    const Mat& img, Ptr<FeatureDetector>& detector,
    vector<KeyPoint>& keypoints, Mat& descriptors) {

    keypoints.clear(); 
    detector->detect(img, keypoints);

    //detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    cout << "DETECTED FEATURES num = " << to_string(keypoints.size());
}

void trainOnCurrentFrame(
    Mat& trainedPageImage,
    const Mat& frame, Rect& pageRegion, 
    Ptr<FeatureDetector>& featureDetector,
    vector<KeyPoint>& trainedKeypoints, Mat& trainedDescriptors) {
    
    cout << endl << "TRAINING" << endl;
    trainedPageImage = capturePageRegion(frame, pageRegion);
    detectFeatures(trainedPageImage, featureDetector, trainedKeypoints, trainedDescriptors);
}

// Filter KNN matches using Lowe's ratio test
vector<DMatch> filterMatches(const vector<vector<DMatch>>& knn_matches, float ratio = 0.7f) {
    vector<DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) {
            // if the best match (0) is way better than the second best (1) then this is a good overall match
            good_matches.push_back(m[0]);
        }
    }
    return good_matches;
}

// Draw rectangle around page region
void drawPageArea(Mat& frame, const Rect& area, Scalar color) {
    rectangle(frame, area, color, 2);
}

// Save new page image and update descriptors
void saveNewPage(const Mat& frame, const Rect& pageRegion,
    Ptr<FeatureDetector>& featureDetector,
    vector<KeyPoint>& pageKeypoints, Mat& pageDescriptors,
    int& imagesSaved) {
    Mat newPage = capturePageRegion(frame, pageRegion);
    detectFeatures(newPage, featureDetector, pageKeypoints, pageDescriptors);
    imwrite(to_string(imagesSaved++) + ".jpg", newPage);
    cout << "** New page detected and saved: " << imagesSaved - 1 << ".jpg **" << endl;
}


string to_string2(int num, int width = 5, char fill = ' ') {
    string s = to_string(num);
    if (s.length() < width)
        s = string(width - s.length(), fill) + s;
    return s;
}

// text to be displayed on the image frame
String statusText = "";
// Adds to the status text that's already on the screen
void addStatusText(Mat& frame, String appendText) {
    statusText += appendText;
    putText(frame, statusText,
        Point(50, 40), FONT_HERSHEY_SIMPLEX, 0.4, 0);
}

// Grab the next frame at the current keyboardTime
// Returns false if the frame is empty (end of video), true otherwise
bool grabFrame(VideoCapture& cap, long videoTime, Mat& frame, int width = 1024, int height = 768) {
    long msecsNow = videoTime;
    cap.set(CAP_PROP_POS_MSEC, msecsNow);
    cout << msecsNow << "::";

    cap.read(frame);

    if (frame.empty()) {
        cout << "error: videocapture got empty frame" << endl;
        return false; // signal that video ended
    }

    resize(frame, frame, Size(width, height), INTER_LINEAR);
    statusText = "";
    
    addStatusText(frame, "Time: " + std::to_string(msecsNow));
    
    return true;
}

bool areImagesSimilar(const vector<DMatch>& matches,
    const vector<KeyPoint>& kpts1,
    const vector<KeyPoint>& kpts2,
    double& outAvg, 
    double& outStdev,
    double maxAvgDist = 50.0,     // threshold you can adjust
    double maxStdDev = 20.0) {
    if (matches.size() < 4) return false;

    vector<double> distances;
    distances.reserve(matches.size());

    for (const auto& m : matches) {
        Point2f p1 = kpts1[m.queryIdx].pt; // cam frame
        Point2f p2 = kpts2[m.trainIdx].pt; // saved page

        double d = norm(p1 - p2); // Euclidean distance
        distances.push_back(d);
    }

    // Calculate average distance
    double sum = 0;
    for (double d : distances) sum += d;
    double avg = sum / distances.size();

    // Calculate standard deviation
    double variance = 0;
    for (double d : distances) variance += (d - avg) * (d - avg);
    variance /= distances.size();
    double stddev = sqrt(variance);

    outAvg = avg;
    outStdev = stddev;

    return (avg < maxAvgDist&& stddev < maxStdDev);
}


// =======================
// Main Program
// =======================
int main() {
   
    // Load video
    VideoCapture cap("..\\videos\\journaltest1.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open video file\n";
        return -1;
    }

    // Define page region in the video
    Rect pageRegion(50, 50, 800, 700); // x, y, width, height

    // Initialize detector and matcher
    //Ptr<FeatureDetector> featureDetector = SIFT::create();
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    /**
    // Create a FAST detector
    // Arguments: threshold, nonmaxSuppression, type
    Ptr<FeatureDetector> featureDetector = FastFeatureDetector::create(
        25,                 // intensity threshold
        true,               // enable nonmax suppression (recommended)
        FastFeatureDetector::TYPE_9_16  // circle of 16 pixels, using 9 for test
    );
    **/

    Ptr<FeatureDetector> featureDetector = ORB::create();
    //Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING2, false);

    // Training page descriptors
    vector<KeyPoint> trainedKeypoints;
    Mat trainedDescriptors;

    // Variables for live frame
    vector<KeyPoint> camKeypoints;
    Mat camDescriptors;
    Mat frame;

    // GUI flags
    bool gui_showPageArea = true;
    bool gui_showAllCamFeatures = false;

    // Page training / state
    bool initialPageSet = false;
    int imagesSaved = 0;
    //int lastNumMatches = 0;

    
    // TIMING 

    bool autoForward = true; // if true, will not wait for keys
    
    const int skipTime = 500; // ms between frames to skip
    long videoTime = 0; // initial start time; and then used to track current time
    //long startTime = getTimeMS();
    long lastProcessedTime = -1;



    cout << "Press SPACE to set initial page, 'a'/'s' to rewind/forward, '1'/'2' to toggle displays, ESC to quit." << endl;

    Mat trainedPageImage;
    Mat currentMatchedImage;

    vector<vector<DMatch>> knnMatches;
    vector<DMatch> goodMatches;


    bool firstFrame = true;
    bool lastFrameWasSameAsPrev = false;

    // grab first frame
    grabFrame(cap, videoTime, frame);

    while (true) {
        
        
        // ---- Handle keyboard input ----
        char key = (char)waitKey(autoForward ? 1 : 10);
        if (key == 27) break; // ESC
        if (autoForward || key == 's') {
            videoTime += skipTime; if (!grabFrame(cap, videoTime, frame)) break;
        }
        if (key == 'a') { videoTime -= skipTime; if (!grabFrame(cap, videoTime, frame)) break;
        }
        if (key == '1') gui_showPageArea = !gui_showPageArea;
        if (key == '2') gui_showAllCamFeatures = !gui_showAllCamFeatures;
        if (key == 'd') imshow("Current", frame);

        // ---- Train a page ----
        if (key == ' ') {
            trainOnCurrentFrame(trainedPageImage, frame, pageRegion, featureDetector, trainedKeypoints, trainedDescriptors);
            lastProcessedTime = -1;
            cout << "** Initial page set **" << endl;
        }

        // ---- Draw GUI elements ----
        if (gui_showPageArea) drawPageArea(frame, pageRegion, Scalar(0, 255, 0));

        
        if (lastProcessedTime == videoTime) {
            // we didn't move the time, so do nothing
            //imshow("Journal", frame);
            continue;
        }


        
        // ---- Feature detection and page change check ----
        if (true) {
            
            lastProcessedTime = videoTime;
            cout << lastProcessedTime << " ";
            
            currentMatchedImage = capturePageRegion(frame, pageRegion);
            detectFeatures(currentMatchedImage, featureDetector, camKeypoints, camDescriptors);

            
            // Match features

            cout << endl << "MATCHING... " << endl;
            

            // Show all camera features if enabled
            if (gui_showAllCamFeatures) {
                cv::Mat img1_with_kp, img2_with_kp;

                // Draw keypoints on each image separately
                cv::drawKeypoints(currentMatchedImage, camKeypoints, img1_with_kp, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DEFAULT);
                cv::drawKeypoints(trainedPageImage, trainedKeypoints, img2_with_kp, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DEFAULT);

                cv::Mat combined;
                cv::hconcat(img1_with_kp, img2_with_kp, combined);
                cv::imshow("Keypoints", combined);
            
            } 
            if (false) { //gui_showAllCamFeatures) {
                //Mat output;
                //drawKeypoints(pageImage, camKeypoints, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                //imshow("Camera Features", output);


                Mat img_matches;
                drawMatches(
                    currentMatchedImage, camKeypoints,
                    trainedPageImage, trainedKeypoints,
                    goodMatches,
                    img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                imshow("Good Matches", img_matches);

            }

            // Analyze differences

            ImageChecker* checker = new ImageChecker_DensityGrid();
            bool isSamePage = checker->compareKeypointStats(camKeypoints, trainedKeypoints, &currentMatchedImage, &trainedPageImage);

            if (isSamePage) {
                cout << "-> Likely same page.\n";
            }
            else {
                cout << "-> Likely different page.\n";
            }
            
            if (!isSamePage) drawPageArea(frame, pageRegion, Scalar(255, 0, 0));



            // Now smartly track what we've been seeing

            cout << "isSamePage:" << isSamePage << " lastSameAsPrev:" << lastFrameWasSameAsPrev << " FF:" << firstFrame << endl;

            if ((firstFrame) ||
                (isSamePage && !lastFrameWasSameAsPrev)) {

                // we've just stabilized on a new page:
                // is same as the previous frame, but we've been in a period of unmatched pages
                
                // retrain on this because it's usually more stable
                trainOnCurrentFrame(trainedPageImage, frame, pageRegion, featureDetector, trainedKeypoints, trainedDescriptors);
                lastProcessedTime = videoTime;
                
                // TODO: SAVE IT BECAUSE THIS IS GOOD
                cout << endl << "-------------- GOOD PAGE ------------------" << endl;
                cout << "-------------- GOOD PAGE ------------------" << endl;
                
                imwrite(std::to_string(imagesSaved++) + ".jpg", frame);


                drawPageArea(frame, pageRegion, Scalar(0, 0, 255));

                isSamePage = true; // set it so we skip through the next if statement
            }

           
            if (isSamePage) {
                lastFrameWasSameAsPrev = true;

            }
            else {
                // we're going to train on this so we can see when it becomes stable
                trainOnCurrentFrame(trainedPageImage, frame, pageRegion, featureDetector, trainedKeypoints, trainedDescriptors);
                lastProcessedTime = videoTime;

                lastFrameWasSameAsPrev = false;
            }
        }

        firstFrame = false;

        // ---- Show live video ----
        imshow("Journal3", frame);
        cout << "REDRAW" << endl;
    }

    return 0;
}
