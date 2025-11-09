#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// Abstract base class
class ImageChecker {
public:
    
    //virtual void computeKeypointStats(const vector<KeyPoint>& kps) const = 0;
    //virtual void compareKeypointStats(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2) const = 0;
    virtual void compareKeypointStats(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, const cv::Mat* img1 = nullptr, const cv::Mat* img2 = nullptr) const = 0;

    virtual ~ImageChecker() = default; // Always good practice in polymorphic base classes
};

// Derived class implementing centroid-based comparison
class ImageChecker_GlobalCentroid : public ImageChecker {
public:
    struct KeypointStats {
        Point2f mean;
        Point2f stddev;
    };
    
    KeypointStats computeKeypointStats(const vector<KeyPoint>& kps) const {
        if (kps.empty()) return { Point2f(0, 0), Point2f(0, 0) };

        vector<float> xs, ys;
        xs.reserve(kps.size());
        ys.reserve(kps.size());

        for (const auto& kp : kps) {
            xs.push_back(kp.pt.x);
            ys.push_back(kp.pt.y);
        }

        Scalar meanX, meanY, stdX, stdY;
        meanStdDev(xs, meanX, stdX);
        meanStdDev(ys, meanY, stdY);

        return { Point2f(meanX[0], meanY[0]), Point2f(stdX[0], stdY[0]) };
    }

    void compareKeypointStats(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, const cv::Mat* img1 = nullptr, const cv::Mat* img2 = nullptr) const override {
        KeypointStats s1 = computeKeypointStats(kp1);
        KeypointStats s2 = computeKeypointStats(kp2);

        float centroidDist = norm(s1.mean - s2.mean);
        float spreadDiff = norm(s1.stddev - s2.stddev);

        cout << "Image 1: mean=(" << s1.mean.x << ", " << s1.mean.y << ") "
            << "std=(" << s1.stddev.x << ", " << s1.stddev.y << ")\n";
        cout << "Image 2: mean=(" << s2.mean.x << ", " << s2.mean.y << ") "
            << "std=(" << s2.stddev.x << ", " << s2.stddev.y << ")\n";

        cout << "Centroid distance: " << centroidDist << " pixels\n";
        cout << "Spread difference: " << spreadDiff << " pixels\n";

        if (centroidDist < 30 && spreadDiff < 15)
            cout << "-> Likely same page.\n";
        else
            cout << "-> Likely different page.\n";
    }
};

/***

class ImageChecker_TEST : public ImageChecker {
public:

    void compareKeypointStats(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2) const override {
    }
};

****/

class ImageChecker_DensityGrid : public ImageChecker {
public:


    // Compute normalized 2D histogram (keypoint density map)
    cv::Mat keypointDensity(const std::vector<cv::KeyPoint>& keypoints,
        int imgWidth, int imgHeight,
        int gridX = 10, int gridY = 10) const
    {
        cv::Mat hist = cv::Mat::zeros(gridY, gridX, CV_32F);

        // add points to the griddified bins - this essentially makes a histogram
        for (const auto& kp : keypoints) {
            int xbin = int((kp.pt.x / imgWidth) * gridX);
            int ybin = int((kp.pt.y / imgHeight) * gridY);
            if (xbin >= gridX)  xbin = gridX - 1;
            if (ybin >= gridY)  ybin = gridY - 1;
            hist.at<float>(ybin, xbin) += 1.0f;
        }

        // Normalize so total sum = 1
        hist /= (float)keypoints.size();

        return hist;
    }


    void compareKeypointStats(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, const cv::Mat* img1 = nullptr, const cv::Mat* img2 = nullptr) const override {

        // Compute grid histograms
        int gridSize = 10;
        Mat h1 = keypointDensity(kp1, img1->cols, img1->rows, gridSize, gridSize);
        Mat h2 = keypointDensity(kp2, img2->cols, img2->rows, gridSize, gridSize);

        // Compare distributions (Cosine similarity)
        double dotProd = h1.dot(h2);
        double norm1 = norm(h1);
        double norm2 = norm(h2);
        double cosineSim = dotProd / (norm1 * norm2 + 1e-8);

        // You could also use correlation or Chi-square if you prefer:
        // double corr = compareHist(h1, h2, HISTCMP_CORREL);
        // double chi2 = compareHist(h1, h2, HISTCMP_CHISQR);

        cout << "Cosine similarity: " << cosineSim << endl;

        if (cosineSim > 0.7f)
            cout << "-> Likely same page.\n";
        else
            cout << "-> Likely different page.\n";


        // Optional: visualize histograms as heatmaps
        Mat vis1, vis2;
        normalize(h1, vis1, 0, 255, NORM_MINMAX);
        normalize(h2, vis2, 0, 255, NORM_MINMAX);
        vis1.convertTo(vis1, CV_8U);
        vis2.convertTo(vis2, CV_8U);
        resize(vis1, vis1, Size(100, 100));
        resize(vis2, vis2, Size(100, 100));
        imshow("Density1", vis1);
        imshow("Density2", vis2);
    }
};



