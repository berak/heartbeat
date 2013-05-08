#include "opencv2/core.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp" 
#include <iostream>
using namespace cv;
using namespace std;



template <typename Vec, typename Operator>
void foreach(Vec &v, Operator &op,size_t off,size_t len) {
    for (size_t i=off; i<len; ++i) {
        op(i,v[i]);
    }
}

template <typename Vec, typename Operator>
void foreach(Vec &v, Operator &op) {
    return foreach(v,op,0,v.size());
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
        if ( it > M ) {  Mi = i;  M = it;  }
        if ( it < m ) {  mi = i;  m = it;  }
    }
};


inline
double hamming(double n,int N) 
{ 
    return 0.54-0.46*cos((2.0*CV_PI*n)/(N-1)); 
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
        p += 1; 
        p %= elm.size();

        elm[p]=v; 
        tim[p]=t;         
    }

    int next()
    {
        return (p+1)%elm.size();
    }
    int last()
    {
        return (p-1)%elm.size();
    }
    // our timebuffer got sampled, whenever there was a frame available.
    // upsample to len samples evenly spaced in time 
    int wrap( vector<float> & din, size_t len=512, float ts=0.02f )
    {
        int   e  = next();
        float t  = float(tim[e]/getTickFrequency());
        float tz = float(tim[p]/getTickFrequency());
        while( din.size()<len && t<tz )
        {
            int  nxt = (e+1)%elm.size();
            float t0 = float(tim[e]/getTickFrequency());
            float t1 = float(tim[nxt]/getTickFrequency());
            float v0 = elm[e];
            float v1 = elm[nxt];
            float v  = float(v0+(t-t0)*(v1-v0)/(t1-t0)); // lerp
            din.push_back(v);

            t += ts;
            if ( t >= t1 )
            {
                e = nxt;
            }
        }
        return p-e;
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
    bool doHam=true;
    bool doAbs=true;

    int hind = 2;
    int spike_thresh=30;
    int tpos=15;
    int twid=14;
    int rsize=256;
    Ring ring(rsize);
    Ring peak(128);
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
        Scalar m  = mean(frame(region));
        Scalar m2 = mean(frame(region_2));
        float z = float(m[hind]-m2[hind]);
        ring.push(t0, z);

        rectangle(frame,region,Scalar(200,0,0));
        line(frame,region.tl(),Point(region.tl().x,region.tl().y+int(m[0]/4)),Scalar(150,170,0),5);
        rectangle(frame,region_2,Scalar(20,80,10));
        line(frame,region_2.tl(),Point(region_2.tl().x,region_2.tl().y+int(m2[0]/4)),Scalar(150,170,0),5);

        int disiz=0,dosiz=0,Mi=0;
        float pf = 1.0f;
        while ( doDct && ring.elm.back() != 0 ) // once
        {
            pf = 3.0f;
            // skip dft if input contains spikes
            MinMaxId mm;
            foreach(ring.elm,mm);
            if (mm.M-mm.m > spike_thresh)
                break;

            vector<float> din, dout;
            int left = ring.wrap(din);
            disiz=ring.elm.size()-left;
            dosiz=din.size();
            if ( doHam )
                foreach(din,Hamming(din.size()));

            dft( din, dout );

            if ( doAbs )
                foreach(dout,Abs);
            paint(frame,dout,50,250,Scalar(0,0,200),1, 1.0f);
            rectangle(frame,Point(50+tpos,250-30),Point(50+tpos+twid,250+30),Scalar(200,0,0));

            MinMaxId mm2;
            foreach(dout,mm2,20,dout.size());
            rectangle(frame,Point(50+mm2.Mi-5,250-30),Point(50+mm2.Mi+5,250+30),Scalar(2,100,0));
            Mi=mm2.Mi;

            // process the selected fft window:
            vector<float> clipped;
            clipped.insert(clipped.begin(),dout.begin()+tpos,dout.begin()+tpos+twid);

            vector<float> idft;
            dft(clipped,idft,DFT_INVERSE);

            if ( doHam )
                foreach(idft,Hamming(idft.size()));
            paint(frame,idft,50+tpos,250-30,Scalar(0,220,220),0.8f,5.0f);

            // peak of idft is the actual cardiac
            MinMaxId mm3;
            foreach(idft,mm3);
            peak.push(mm3.Mi,mm3.M);
            paint(frame,peak.elm,50,350,Scalar(0,80,0),1,4);
            circle(frame,Point(50+peak.p*4,350+int(peak.elm[peak.p])),3,Scalar(60,230,0),2);

            break;
        }
        paint(frame,ring.elm,50,50-int(z),Scalar((hind==0?200:0),(hind==1?200:0),(hind==2?200:0)),pf);
        circle(frame,Point(50+ring.p,50-int(z)+int(pf*ring.elm[ring.p])),3,Scalar(60,230,0),2);
        imshow("cam",frame);
        int k = waitKey(1);
        if ( k==27 ) break;
        if ( k=='1') doDct=!doDct;
        if ( k=='2') doHam=!doHam;
        if ( k=='3') doAbs=!doAbs;
        if ( k=='4') { hind++; hind%=3; cerr << "hind " << hind << endl; }
        int64 t1 = getTickCount();
        t += (t1-t0);
        if ( f % 10 ==9 )
        {
            double fps =  1.0 / double( (t/10) /getTickFrequency());
            cerr << format("%4d %3.3f %3.3f %6d %6d %6d",f,fps,z,disiz,dosiz,Mi) << endl;
            t = 0;
        }
        f ++;
    }
    return 0;
}

