#include "opencv2/core.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp" 
#include <iostream>
using namespace cv;
using namespace std;


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

template <typename Vec, typename Operator>
void foreach(Vec &v, Operator op)
{
    Vec::iterator it = v.begin();
    for (; it!=v.end(); ++it)
    {
        op(*it);
    }
}
template <typename Vec, typename Operator>
void foreach_i(Vec &v, Operator op)
{
    Vec::iterator it = v.begin();
    for (size_t i=0; i<v.size(); ++i)
    {
        op(i,v[i]);
    }
}
template <typename Vec, typename Operator>
void Where(const Vec &a, Vec &b, Operator op)
{
    Vec::iterator it = a.begin();
    for (; it!=a.end(); ++it)
    {
        if ( op(*it) )
            b.push_back(*it);
    }
}

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
//void hamming(vector<float> & v)
//{
//    size_t vs = v.size();
//    for ( size_t i=0; i<vs; i++ )
//    {
//        v[i] = v[i] * float(hamming(i,vs));
//    }
//}

struct Ring
{
    vector<float> elm;
    int p;

    Ring(int n=0) : elm(n), p(0)  {}

    operator float & ()      { return elm[p]; }
    float & operator ++(int) { p += 1; p %= elm.size(); return elm[p]; } // postfix

    // wrap timebuffer around current pos:
    void wrap( vector<float> & din)
    {
        din.insert( din.begin(), elm.begin()+p, elm.end() );
        din.insert( din.end(), elm.begin(), elm.begin()+p );
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

    int rsize=240;
    Ring ring(rsize);

    Rect region = Rect(130,100,60,60);

    namedWindow("cam",0);
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
        ring ++ = float(m[0]);
        if ( doDct )
        {
            vector<float> dout;
            vector<float> din;
            ring.wrap(din);

            if ( doHam )
                foreach_i(din,Hamming(din.size()));

            dft( din, dout );

            foreach(dout,Abs);
            paint(frame,dout,50,250,Scalar(0,0,200),1);

            int bpm_limits[] = {20,60};
            vector<float> clipped;
            clipped.insert(clipped.begin(),dout.begin()+bpm_limits[0],dout.begin()+bpm_limits[1]);

            vector<float> idft;
            dft(clipped,idft,DFT_INVERSE);
            //foreach_i(idft,HammingInv(idft.size()));
            paint(frame,idft,350,50,Scalar(0,220,220),0.25f);
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
            cerr << format("%4d %3.3f %3.3f",f,fps,float(m[0])) << endl;
            t = 0;
        }
        f ++;
    }
    return 0;
}

