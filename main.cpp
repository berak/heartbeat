#include "opencv2/core.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp" 
#include <iostream>
using namespace cv;
using namespace std;



template <typename Vec, typename Operator>
void foreach(Vec &v, Operator &op) {
    for (size_t i=0; i<v.size(); ++i) {
        op(i,v[i]);
    }
}


void Abs(int i, float &it)
{
    it = abs(it);
}

struct MinMaxId
{
    float m,M;
    int mi,Mi;
    MinMaxId() : m(99999),M(-99999),mi(0),Mi(0) {}
    void operator () (int i, float & it)
    {
        if ( it > M )
        {
            Mi = i;
            M = it;
        }
        if ( it < m )
        {
            mi = i;
            m = it;
        }
    }
};


inline
double hamming(double n,int N) 
{ 
    const double twopi = 2.0 * CV_PI;// 24.0/7.0;
    return 0.54-0.46*cos((twopi*n)/(N-1)); 
}
struct Hamming
{
    int N;
    Hamming(int n) : N(n) {}
    void operator () (int i, float & it)
    {
        it = it * float(hamming(i,N));
    }
};
struct HammingInv
{
    int N;
    HammingInv(int n) : N(n) {}
    void operator () (int i, float & it)
    {
        it = it / float(hamming(i,N));
    }
};


struct Ring
{
    vector<float> elm;
    vector<int64> tim;
    int p;

    Ring(int n=1) : elm(n,0),tim(n,0), p(0)  {}

    void push(int64 t, float v) 
    { 
        elm[p]=v; 
        tim[p]=t; 
        
        p += 1; 
        p %= elm.size();
    }

    // wrap timebuffer around current pos and interpolate:
    void _wrap( vector<float> & din, int i, float ts=0.02f, int maxdin=512 )
    {
        int next = (i+1)%elm.size();

        int64 t0 = tim[i];
        if ( t0==0 ) return;
        int64 t1 = tim[next];
        if ( t1<=t0 ) return;
        double dt = (t1-t0) / getTickFrequency();

        double v0 = elm[i];
        double v1 = elm[next];
        double ds = (v1-v0);

        for ( float t=0; t<dt; t+=ts )
        {
            if ( din.size() >= maxdin )
                return;
            float v = float(v0+t*ds/dt); // lerp
            din.push_back(v);
        }
    }

    void wrap( vector<float> & din )
    {
        int maxdin=512;
        float ts = 0.02f;
        for( size_t i=p+1; i<elm.size(); i++ )
        {
            _wrap(din,i,ts,maxdin);
        }
        for( size_t i=0; i<p; i++ )
        {
            _wrap(din,i,ts,maxdin);
        }
    }
};

void paint(Mat & img, const vector<float> & elm, int x, int y, Scalar col,float sy=1.0f, float sx=1.0f)
{
    for ( size_t i=1; i<elm.size(); i++ )
    {
        line(img,Point(x+int(sx*(i-1)),y+int(sy*elm[i-1])),Point(x+int(sx*(i)),y+int(sy*elm[i])),col,2);
    }
}

int main( int argc, char** argv )
{
    bool doDct=true;
    bool doHam=false;

    int spike_thresh=30;
    int tpos=50;
    int twid=20;
    int rsize=256;
    Ring ring(rsize);

    Rect region = Rect(130,100,60,60);
    Rect region_2 = Rect(440,100,60,60);

    namedWindow("cam",0);
    namedWindow("control",0);
    createTrackbar("pos","control",&tpos,512-128);
    createTrackbar("width","control",&twid,128);
    VideoCapture cap(0);
    // cap.set(CAP_PROP_SETTINGS,1);
    int f = 0;
    int64 t = 0;
    while( cap.isOpened() )
    {
        int64 t0 = getTickCount();
        Mat frame;
        cap >> frame;
        if ( frame.empty() )
            break;
        
        // take the diff of a skin to a non-skin rect as measure for change
        Mat roi2 = frame(region_2);
        Mat rgb2[3]; split(roi2, rgb2);
        Scalar m2 = mean(rgb2[2]);

        Mat roi = frame(region);
        Mat rgb[3]; split(roi, rgb);
        Scalar m = mean(rgb[2]);

        float z = float(m[0]-m2[0]);
        ring.push(t0, z);

        rectangle(frame,region,Scalar(200,0,0));
        rectangle(frame,region_2,Scalar(50,150,0));
        line(frame,region.tl(),Point(region.tl().x,region.tl().y+m[0]/4),Scalar(150,170,0),6);
        line(frame,region_2.tl(),Point(region_2.tl().x,region_2.tl().y+m2[0]/4),Scalar(150,170,0),6);
        int disiz=0,dosiz=0;
        while ( doDct && ring.elm.back() != 0 ) // once
        {
            // check for spikes
            MinMaxId mm;
            foreach(ring.elm,mm);
            if (mm.M-mm.m > spike_thresh)
                break;

            vector<float> din;
            vector<float> dout;
            ring.wrap(din);
            disiz=ring.elm.size();
            dosiz=din.size();
            foreach(din,Hamming(din.size()));

            dft( din, dout );

            foreach(dout,Abs);
            paint(frame,dout,50,250,Scalar(0,0,200),1, 1);
            rectangle(frame,Point(50+tpos,250-30),Point(50+tpos+twid,250+30),Scalar(200,0,0));

            // process the selected fft window:
            vector<float> clipped;
            clipped.insert(clipped.begin(),dout.begin()+tpos,dout.begin()+tpos+twid);

            vector<float> idft;
            dft(clipped,idft,DFT_INVERSE);

            foreach(idft,Hamming(idft.size()));
            paint(frame,idft,50+tpos,250-30,Scalar(0,220,220),0.8f,5.0f);
            break;
        }
        float pf = 1.0f;
        paint(frame,ring.elm,50,50,Scalar(0,200,0),pf);
        int p = (ring.p-1+ring.elm.size())%ring.elm.size();
        circle(frame,Point(50+p,50+int(pf*ring.elm[p])),3,Scalar(60,230,0),2);
        imshow("cam",frame);
        int k = waitKey(1);
        if ( k==27 ) break;
        if ( k=='1') doDct=!doDct;
        if ( k=='2') doHam=!doHam;
        int64 t1 = getTickCount();
        t += (t1-t0);
        if ( f % 10 ==9 )
        {
            double fps =  1.0 / double( (t/10) /getTickFrequency());
            cerr << format("%4d %3.3f %3.3f %6d %6d",f,fps,z,disiz,dosiz) << endl;
            t = 0;
        }
        f ++;
    }
    return 0;
}

