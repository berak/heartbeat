#include "opencv2/core.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp" 
#include <iostream>
using namespace cv;
using namespace std;



template <typename Vec, typename Operator>
void foreach(Vec &v, Operator op) {
    Vec::iterator it = v.begin();
    for (; it!=v.end(); ++it) {
        op(*it);
    }
}

template <typename Vec, typename Operator>
void foreach_i(Vec &v, Operator op) {
    for (size_t i=0; i<v.size(); ++i) {
        op(i,v[i]);
    }
}


void Abs(float &it)
{
    it = abs(it);
}

struct MaxId
{
    int id;
    float m;
    MaxId() : id(0),m(-99999) {}
    void operator () (int i, float & it)
    {
        if ( it > m )
        {
            id = i;
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

inline float ipol(float a,float b,float t,float dt)
{
    return (a*(1.0-t) + b*t) / dt;
}


struct Ring
{
    vector<float> elm;
    vector<int64> tim;
    int p;

    Ring(int n=1) : elm(n),tim(n), p(0)  {}

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
            float v = float(v0+t*ds/dt);
            din.push_back(v);
        }
    }

    void wrap( vector<float> & din )
    {
        for( size_t i=p+1; i<elm.size(); i++ )
        {
            _wrap(din,i);
        }
        for( size_t i=0; i<p; i++ )
        {
            _wrap(din,i);
        }
    }
};

void paint(Mat & img, const vector<float> & elm, int x, int y, Scalar col,float s=1.0f)
{
    for ( size_t i=1; i<elm.size(); i++ )
    {
        line(img,Point(x+i-1,y+int(s*elm[i-1])),Point(x+i,y+int(s*elm[i])),col,2);
    }
}

int main( int argc, char** argv )
{
    bool doDct=false;
    bool doHam=false;

    int tpos=50;
    int twid=20;
    int rsize=240;
    Ring ring(rsize);

    Rect region = Rect(130,100,60,60);

    namedWindow("cam",0);
    namedWindow("control",0);
    createTrackbar("pos","control",&tpos,512-128);
    createTrackbar("wid","control",&twid,128);
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
        rectangle(frame,region,Scalar(200,0,0));
        Mat roi = frame(region);
        Mat rgb[3];
        split(roi, rgb);
        Scalar m = mean(rgb[2]);
        ring.push(t0, float(m[0]));
        int disiz=0,dosiz=0;
        if ( doDct )
        {
            vector<float> din;
            vector<float> dout;
            ring.wrap(din);
            disiz=ring.elm.size();
            dosiz=din.size();
            if ( doHam )
                foreach_i(din,Hamming(din.size()));

            dft( din, dout );

            foreach(dout,Abs);
            paint(frame,dout,50,250,Scalar(0,0,200),1);
            rectangle(frame,Point(50+tpos,250-30),Point(50+tpos+twid,250+30),Scalar(200,0,0));

            vector<float> clipped;
            clipped.insert(clipped.begin(),dout.begin()+tpos,dout.begin()+tpos+twid);

            vector<float> idft;
            dft(clipped,idft,DFT_INVERSE);
            foreach_i(idft,Hamming(idft.size()));
            paint(frame,idft,350,50,Scalar(0,220,220),0.5f);
        }
        paint(frame,ring.elm,50,50,Scalar(0,200,0),2);
        circle(frame,Point(50+ring.p,50+int(2*ring.elm[ring.p])),3,Scalar(60,230,0),2);
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
            cerr << format("%4d %3.3f %3.3f %6d %6d",f,fps,float(m[0]),disiz,dosiz) << endl;
            t = 0;
        }
        f ++;
    }
    return 0;
}

