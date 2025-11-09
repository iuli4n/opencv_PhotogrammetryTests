/******
JOURNAL PAGE EXTRACTOR FROM VIDEO OF FLIPPING PAGES

The way it's supopsed to work:
- detect a clean page and train a set of keypoints on it
- as the next frames go by, see if the next frames are of the same page or not (based on keypoints check)
- if detected new page, save the old one and train on the new one


CURRENT ISSUES
- let's say you have two frames of the same page. the keypoints are totally off between them. 
i tried changing the detectmatches from FLANN to BRUTEFORCE but the system just totally breaks [i think the knnMatcher gives bad results]




*******/

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

// Detect features and compute descriptors
void detectFeatures(const Mat& img, Ptr<FeatureDetector>& detector,
    vector<KeyPoint>& keypoints, Mat& descriptors) {

    keypoints.clear(); 
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    cout << "DETECTED FEATURES num = " << to_string(keypoints.size());
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

// Crop and convert page region to grayscale
Mat capturePageRegion(const Mat& frame, const Rect& area) {
    Mat cropped = frame(area);//.clone();
    Mat gray;
    cvtColor(cropped, gray, COLOR_RGB2GRAY);
    return gray;
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
bool grabFrame(VideoCapture& cap, long keyboardTime, Mat& frame, int width = 1024, int height = 768) {
    long msecsNow = keyboardTime;
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

    Ptr<FeatureDetector> featureDetector = ORB::create();
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING2, false);

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
    int lastNumMatches = 0;

    // Keyboard time simulation
    long keyboardTime = 1000;
    long startTime = getTimeMS();
    long lastProcessedTime = -1;

    cout << "Press SPACE to set initial page, 'a'/'s' to rewind/forward, '1'/'2' to toggle displays, ESC to quit." << endl;

    Mat trainedPageImage;
    Mat currentMatchedImage;

    vector<vector<DMatch>> knnMatches;
    vector<DMatch> goodMatches;



    // grab first frame
    grabFrame(cap, keyboardTime, frame);

    while (true) {
        
        

        // ---- Handle keyboard input ----
        char key = (char)waitKey(10);
        if (key == 27) break; // ESC
        if (key == 'a') { keyboardTime -= 1000; if (!grabFrame(cap, keyboardTime, frame)) break; 
        }
        if (key == 's') { keyboardTime += 1000; if (!grabFrame(cap, keyboardTime, frame)) break; 
        }
        if (key == '1') gui_showPageArea = !gui_showPageArea;
        if (key == '2') gui_showAllCamFeatures = !gui_showAllCamFeatures;
        if (key == 'd') imshow("Current", frame);

        // ---- Train a page ----
        if (key == ' ') {
            trainedPageImage = capturePageRegion(frame, pageRegion);
            detectFeatures(trainedPageImage, featureDetector, trainedKeypoints, trainedDescriptors);
            initialPageSet = true;
            lastNumMatches = 0;
            lastProcessedTime = -1;
            cout << "** Initial page set **" << endl;
        }

        // ---- Draw GUI elements ----
        if (gui_showPageArea) drawPageArea(frame, pageRegion, Scalar(0, 255, 0));

        
        if (lastProcessedTime == keyboardTime) {
            // we didn't move the time, so do nothing
            //imshow("Journal", frame);
            continue;
        }

        // ---- Feature detection and page change check ----
        if (initialPageSet) {
            
            lastProcessedTime = keyboardTime;
            cout << lastProcessedTime << " ";
            
            currentMatchedImage = capturePageRegion(frame, pageRegion);
            detectFeatures(currentMatchedImage, featureDetector, camKeypoints, camDescriptors);

            
            // Match features

            cout << endl << "MATCHING... " << endl;
            cout << "TRAIN type " << trainedDescriptors.type() << " size " << trainedDescriptors.size() << "  sum " << cv::sum(trainedDescriptors.row(0)) << std::endl;
            cout << "CAM type " << camDescriptors.type() << " size " << camDescriptors.size() << "  sum " << cv::sum(camDescriptors.row(0)) << std::endl;

            matcher->knnMatch(camDescriptors, trainedDescriptors, knnMatches, 2);
            goodMatches = filterMatches(knnMatches, 0.5F);

            // check the results
            for (const auto& match : goodMatches) {
                if (match.queryIdx >= camKeypoints.size()) {
                    std::cerr << "Invalid match indices found! "
                        << match.queryIdx << " >= keypoints size " << camKeypoints.size() << endl;
                }
                if (match.trainIdx >= trainedKeypoints.size()) {
                    std::cerr << "Invalid match indices found! "
                        << match.trainIdx << " >= trained size " << trainedKeypoints.size() << std::endl;
                }

            }


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
            
                ImageChecker* checker = new ImageChecker_DensityGrid();
                checker->compareKeypointStats(camKeypoints, trainedKeypoints, &currentMatchedImage, &trainedPageImage);
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

            double avg, std;
            areImagesSimilar(goodMatches, camKeypoints, trainedKeypoints, avg, std);

            int numGoodMatches = goodMatches.size();
            float deltaPerc = numGoodMatches ? (lastNumMatches - numGoodMatches) * 100.0f / numGoodMatches : 0;

            addStatusText(frame, 
                "  pointsCam=" + to_string2(camKeypoints.size()) + "  pointsPage=" + to_string2(trainedKeypoints.size()) +
                "  matches=" + to_string2(numGoodMatches) +
                "  avgDist=" + to_string2(avg)+ "  std=" +to_string2(std) +
                "  lastNM=" + to_string2(lastNumMatches) + "  deltaPerc=" + to_string2(deltaPerc));

            // Check stability and low match count for page change
            bool stable = abs(deltaPerc) < 20;
            bool lowPoints = numGoodMatches < 10;

            cout << "checked" << endl;

            if (false) {
                if (stable && lowPoints) {
                    cout << "************* [[ NOW ]] ********" << endl;
                    saveNewPage(frame, pageRegion, featureDetector, trainedKeypoints, trainedDescriptors, imagesSaved);
                    lastNumMatches = 500; // reset after saving new page
                }
                else {
                    lastNumMatches = numGoodMatches;
                }
            }
        }

        // ---- Show live video ----
        imshow("Journal3", frame);
        cout << "REDRAW" << endl;
    }

    return 0;
}
