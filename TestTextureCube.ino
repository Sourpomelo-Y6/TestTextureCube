#include <Eigen30.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>
using namespace Eigen;
#include <M5Core2.h>

#include "gimp_image.h"
#include "surface00.h"
#include "surface01.h"
#include "surface02.h"
#include "surface03.h"
#include "surface04.h"
#include "surface05.h"

#define LGFX_AUTODETECT
#include <LovyanGFX.hpp>

static LGFX lcd;
static LGFX_Sprite sprite[2];
static LGFX_Sprite sprite_surface[12];

//#pragma GCC optimize ("O3")
struct point3df{ float x, y, z;};
struct surface{ uint8_t p[4]; int16_t z;const struct gimp_image* pImage; LGFX_Sprite* sprite[2];};
#define U  70     
#define UD  26

int color_array[8]={
  TFT_WHITE,
  TFT_RED,
  TFT_ORANGE,
  TFT_PINK,
  TFT_PURPLE,
  TFT_BLUE,
  TFT_GREEN,
  TFT_SKYBLUE,
};

struct point3df cubef[8] ={ // cube edge length is 2*U
  {  U, -U, UD },//0
  { -U, -U, UD },//1
  { -U, -U, -UD },//2-
  {  U, -U, -UD },//3-
  {  U,  U, UD },//4
  { -U,  U, UD },//5
  { -U,   U, -UD },//6-
  {  U,   U, -UD },//7-
};
 
//struct surface s[6] = {// define the surfaces
//  { {2, 3, 0, 1}, 0 ,&surface01,{0,0}}, // bottom0 right
//  { {7, 6, 5, 4}, 0 ,&surface02,{0,0}}, // top0 left
//  { {4, 0, 1, 5}, 0 ,&surface05,{0,0}}, // back0
//  { {3, 7, 6, 2}, 0 ,&surface00,{0,0}}, // front0
//  { {6, 2, 1, 5}, 0 ,&surface03,{0,0}}, // right1 bottom
//  { {3, 7, 4, 0}, 0 ,&surface04,{0,0}}, // left1 top
//};

struct surface s[6] = {// define the surfaces
  { {2, 3, 0, 1}, 0 ,&surface03,{0,0}}, // bottom
  { {7, 6, 5, 4}, 0 ,&surface04,{0,0}}, // top
  { {4, 5, 1, 0}, 0 ,&surface05,{0,0}}, // back
  { {6, 7, 3, 2}, 0 ,&surface00,{0,0}}, // front
  { {6, 2, 1, 5}, 0 ,&surface02,{0,0}}, // right
  { {3, 7, 4, 0}, 0 ,&surface01,{0,0}}, // left
};

struct point3df cubef2[8];

double pitch = 0.0F;
double roll  = 0.0F;
double yaw   = 0.0F;

bool flip;
uint32_t pre_show_time = 0;
unsigned int pre_time = 0;

int ws;
int hs;

void rotate_cube_xyz( float roll, float pitch, float yaw){
  uint8_t i;

  float DistanceCamera = 300;
  float DistanceScreen = 550;

  float cosyaw   = cos(yaw);
  float sinyaw   = sin(yaw);
  float cospitch = cos(pitch);
  float sinpitch = sin(pitch);
  float cosroll  = cos(roll);
  float sinroll  = sin(roll);

  float sinyaw_sinroll = sinyaw * sinroll;
  float sinyaw_cosroll = sinyaw * cosroll;
  float cosyaw_sinroll = cosyaw * sinroll;
  float cosyaw_cosroll = cosyaw * cosroll;

  float x_x = cosyaw * cospitch;
  float x_y = cosyaw_sinroll * sinpitch - sinyaw_cosroll;
  float x_z = cosyaw_cosroll * sinpitch + sinyaw_sinroll;

  float y_x = sinyaw * cospitch;
  float y_y = sinyaw_sinroll * sinpitch + cosyaw_cosroll;
  float y_z = sinyaw_cosroll * sinpitch - cosyaw_sinroll;

  float z_x = -sinpitch;
  float z_y = cospitch * sinroll;
  float z_z = cospitch * cosroll;

  for (i = 0; i < 8; i++){
    float x = x_x * cubef[i].x
            + x_y * cubef[i].y
            + x_z * cubef[i].z;
    float y = y_x * cubef[i].x
            + y_y * cubef[i].y
            + y_z * cubef[i].z;
    float z = z_x * cubef[i].x
            + z_y * cubef[i].y
            + z_z * cubef[i].z;

    cubef2[i].x = (x * DistanceCamera) / (z + DistanceCamera + DistanceScreen) + (ws>>1);
    cubef2[i].y = (y * DistanceCamera) / (z + DistanceCamera + DistanceScreen) + (hs>>1);
    cubef2[i].z = z;
  }
}

void setup(void){ 
  M5.begin();

  lcd.init();

  lcd.setRotation(1);


// バックライトの輝度を 0～255 の範囲で設定します。
//  lcd.setBrightness(80);

  lcd.setColorDepth(16);  // RGB565の16ビットに設定

  lcd.fillScreen(0);

  lcd.startWrite();
  lcd.fillScreen(TFT_RED);
  lcd.endWrite();

  //ws = lcd.width();
  //hs = lcd.height();
  ws = 160;
  hs = 160;
  
  for (int i = 0; i < 2; i++)
  {
    sprite[i].createSprite(ws,hs);
  }

  for (int i = 0; i < 6; i++)
  {
    sprite_surface[2*i].createSprite(s[i].pImage->width ,s[i].pImage->height);
    sprite_surface[2*i].pushImage(  0, 0, s[i].pImage->width, s[i].pImage->height, (lgfx:: rgb565_t*)s[i].pImage->pixel_data);
    sprite_surface[2*i].setColor(lcd.color565(0,0,0));
    sprite_surface[2*i].fillTriangle(0, 0, 0, s[i].pImage->height-1, s[i].pImage->width-1, s[i].pImage->height-1);
    //sprite_surface[2*i].fillTriangle(0, 0, s[i].pImage->width-1, 0, 0,s[i].pImage->height-1);
  
    sprite_surface[2*i+1].createSprite(s[i].pImage->width ,s[i].pImage->height);
    sprite_surface[2*i+1].pushImage(  0, 0,s[i].pImage->width ,s[i].pImage->height , (lgfx:: rgb565_t*)s[i].pImage->pixel_data);
    sprite_surface[2*i+1].setColor(lcd.color565(0,0,0));
    sprite_surface[2*i+1].fillTriangle(0, 0, s[i].pImage->width-1, s[i].pImage->height-1, s[i].pImage->width-1,0);
    //sprite_surface[2*i+1].fillTriangle(s[i].pImage->width-1, 0, 0, s[i].pImage->height-1, s[i].pImage->width-1, s[i].pImage->height-1);
    
    s[i].sprite[0]=&sprite_surface[2*i];
    s[i].sprite[1]=&sprite_surface[2*i+1];
  }
  
  lcd.startWrite();
  lcd.fillScreen(TFT_DARKGREY);
  lcd.endWrite();
}

void loop() {
  char sensor_string_buff[128];
//  float quat_w, quat_x, quat_y, quat_z;
//  
//  rotate_cube_quaternion(quat_w, quat_x, quat_y, quat_z);
  std::int32_t tx, ty, tc;
  tc = lcd.getTouch(&tx, &ty);
  if (tc)
  {
    roll += (tx > lcd.width()*2/3) ? PI/180.0 : (tx < lcd.width()/3) ? -PI/180.0 : 0;
    pitch += (ty > lcd.height()*2/3) ? PI/180.0 : (ty < lcd.height()/3) ? -PI/180.0 : 0;
  }
  
  rotate_cube_xyz(roll,pitch,yaw);
  
  //描写する面の順番に並び替え
  int ss[6]={0,1,2,3,4,5};
  float sf[6]={0};
  for (int i = 0; i < 6; i++)
  {
    float wz = 0;
    for(int j=0;j<4;j++){
      wz += cubef2[s[i].p[j]].z;
    }
    sf[i] = wz;
  }
  //交換ソート
  for (int j = 5; j > 0; j--){
    for (int i = 0; i < j; i++)
    {
        if(sf[i] < sf[i+1])
        {
          float work = sf[i];
          sf[i] = sf[i+1];
          sf[i+1] = work;
          
          int iw = ss[i];
          ss[i] = ss[i+1];
          ss[i+1] = iw;
        }
    }
  }

  flip = !flip;
  lcd.startWrite();
  sprite[flip].clear();
   //if(show_time > 100)
  {
    for (int i = 0; i < 8; i++)
    {
      //sprite[flip].drawRect( (int)cubef2[i].x-2, (int)cubef2[i].y-2, 4, 4 , 0xF000);
      sprite[flip].drawRect( (int)cubef2[i].y-2, (int)cubef2[i].x-2, 4, 4 , color_array[i]);
      //Serial.printf("%d,%f,%f,\r\n",i,cubef2[i].x, cubef2[i].y); 
    }
    
    //lcd.fillRect( 0, 0, ws, hs   , 0);
    
    for (int i = 3; i < 6; i++)
    {
      int ii = ss[i];
      draw_surface(ii,flip);
    }
  }

  sprite[flip].pushSprite(&lcd, 0, 0);
  
  int show_time = millis() - pre_show_time;
  pre_show_time = millis();
  sprite[flip].setCursor(0, 50);
  sprite[flip].printf("%5d\n",show_time);

  //QuaternionToEulerAngles((double)quat_w, (double)quat_x, (double)quat_y, (double)quat_z, roll, pitch, yaw);
  sprite[flip].setCursor(0, 70);
  sprite[flip].printf("%3.2f\n",roll*180.0/PI);
  sprite[flip].setCursor(0, 90);
  sprite[flip].printf("%3.2f\n",pitch*180.0/PI);
  sprite[flip].setCursor(0, 110);
  sprite[flip].printf("%3.2f\n",yaw*180.0/PI);
  
  sprite[flip].pushSprite(&lcd, 0, 0);
  lcd.endWrite();

}

void draw_surface(int ii, bool flip)
{
 {
    Eigen::MatrixXf tp(3,3);
    tp << cubef2[s[ii].p[0]].y,cubef2[s[ii].p[1]].y,cubef2[s[ii].p[2]].y,
          cubef2[s[ii].p[0]].x,cubef2[s[ii].p[1]].x,cubef2[s[ii].p[2]].x,
            1,  1,  1;
  
    Eigen::MatrixXf fp(3,3);
    fp << 0, s[ii].pImage->width, s[ii].pImage->width,
          0, 0, s[ii].pImage->height,
          1,   1,   1;
  
    Eigen::MatrixXf H(3,3);
    Haffine_from_points(fp,tp,H);
  
    float matrix[6]={
      (float)H(0,0),(float)H(0,1),(float)H(0,2),
      (float)H(1,0),(float)H(1,1),(float)H(1,2)
    };
    s[ii].sprite[0]->pushAffine(&sprite[flip], matrix, 0);
  }

  {
    Eigen::MatrixXf tp(3,3);
    tp << cubef2[s[ii].p[0]].y,cubef2[s[ii].p[2]].y,cubef2[s[ii].p[3]].y,
          cubef2[s[ii].p[0]].x,cubef2[s[ii].p[2]].x,cubef2[s[ii].p[3]].x,
            1,  1,  1;
  
    Eigen::MatrixXf fp(3,3);
    fp << 0, s[ii].pImage->width, 0,
          0, s[ii].pImage->height, s[ii].pImage->height,
          1,   1,   1;
  
    Eigen::MatrixXf H(3,3);
    Haffine_from_points(fp,tp,H);
  
    float matrix[6]={
      (float)H(0,0),(float)H(0,1),(float)H(0,2),
      (float)H(1,0),(float)H(1,1),(float)H(1,2)
    };
    s[ii].sprite[1]->pushAffine(&sprite[flip], matrix, 0);
  }  
}

bool Haffine_from_points(Eigen::MatrixXf& fp, Eigen::MatrixXf& tp, Eigen::MatrixXf& H)
{
    //とりあえず、3x3行列のみを対象にし、形状判断を行わない。
    //if fp.shape != tp.shape:
    //  raise RuntimeError('number of points do not match')

    //# 点を調整する
    //# 開始点
    //m = mean(fp[:2], axis=1)
    Eigen::MatrixXf wfp = fp.topRows(2);
    Eigen::VectorXf m = wfp.rowwise().mean();

    //std::cout << "wfp=\n" << wfp << "\n";
    //std::cout << "m=\n" << m << "\n";
    
    //maxstd = max(std(fp[:2], axis=1)) + 1e-9
    Eigen::VectorXf std_m = (wfp.colwise() - m).array().pow(2).rowwise().mean();
    double maxstd = sqrt(std_m.maxCoeff()) + 1e-9;

    //std::cout << "work=\n" << (wfp.colwise() - m) << "\n";

    //std::cout << "std_m=\n" << std_m << "\n";
    //std::cout << "maxstd=\n" << maxstd << "\n";

    //C1 = diag([1/maxstd, 1/maxstd, 1])
    //C1[0][2] = -m[0]/maxstd
    //C1[1][2] = -m[1]/maxstd
    Eigen::MatrixXf C1(3, 3);
    C1 << 1 / maxstd, 0, -m(0) / maxstd,
        0, 1 / maxstd, -m(1) / maxstd,
        0, 0, 1;

    //std::cout << "C1=\n" << C1 << "\n";

    //fp_cond = dot(C1,fp)
    Eigen::MatrixXf fp_cond = C1 * fp;

    //std::cout << "fp_cond=\n" << fp_cond << "\n";

    //# 対応点
    //m = mean(tp[:2], axis=1)
    //Eigen::MatrixXf wtp = tp(seq(0, last - 1), all);
    Eigen::MatrixXf wtp = tp.topRows(2);
    Eigen::VectorXf m_t = wtp.rowwise().mean();

    //std::cout << "wtp=\n" << wtp << "\n";
    //std::cout << "m_t=\n" << m_t << "\n";

    //C2 = C1.copy()  # 2つの点群で、同じ拡大率を用いる
    //C2[0][2] = -m[0]/maxstd
    //C2[1][2] = -m[1]/maxstd
    Eigen::MatrixXf C2 = C1;
    C2(0, 2) = -m_t(0) / maxstd;
    C2(1, 2) = -m_t(1) / maxstd;

    //std::cout << "C2=\n" << C2 << "\n";

    //tp_cond = dot(C2,tp)
    Eigen::MatrixXf tp_cond = C2 * tp;
    //std::cout << "tp_cond=\n" << tp_cond << "\n";

    Eigen::MatrixXf A(4, 3);
    //A << fp_cond(seq(0, last - 1), all),
    //    tp_cond(seq(0, last - 1), all);
    //std::cout << "Afp~\n" << fp_cond.topRows(fp_cond.cols() - 1) << "\n";
    //std::cout << "Atp~\n" << tp_cond.topRows(tp_cond.cols() - 1) << "\n";

    A << fp_cond.topRows(fp_cond.cols() - 1),
        tp_cond.topRows(tp_cond.cols() - 1);

    //std::cout << "A=\n" << A << "\n";

    //U,S,V = linalg.svd(A.T)
    BDCSVD<MatrixXf> svd(A.transpose(), ComputeFullU | ComputeFullV);

    //# Hartley-Zisserman (第2版) p.130 に基づき行列B,Cを求める
    //tmp = V[:2].T
    Eigen::MatrixXf tmp = svd.matrixV();
    //Eigen::MatrixXf wtmp = tmp(seq(0, 1), all);
    Eigen::MatrixXf wtmp = tmp.leftCols(2);

    //std::cout << "wtmp=\n" << wtmp << "\n";

    //B = tmp[:2]
    //Eigen::MatrixXf B = w2tmp(seq(0, 1), all);
    Eigen::MatrixXf B = wtmp.topRows(2);

    //std::cout << "B=\n" << B << "\n";

    //C = tmp[2:4]
    //Eigen::MatrixXf C = w2tmp(seqN(2, 3), all);
    Eigen::MatrixXf C = wtmp.middleRows(2, 2);

    //std::cout << "C=\n" << C << "\n";

    //tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    Eigen::MatrixXf w = B.completeOrthogonalDecomposition().pseudoInverse();
    w = C * w;
    Eigen::MatrixXf tmp2(2, 3);
    tmp2 << w(0, 0), w(0, 1), 0,
        w(1, 0), w(1, 1), 0;

    //std::cout << "w=\n" << w << "\n";
    //std::cout << "tmp2=\n" << tmp2 << "\n";

    //H = vstack((tmp2,[0,0,1]))
    Eigen::MatrixXf w2(1, 3);
    w2 << 0, 0, 1;
    Eigen::MatrixXf tH(3, 3);
    tH << tmp2,
        w2;

    //std::cout << "w2=\n" << w2 << "\n";
    //std::cout << "tH=\n" << tH << "\n";

    tH = tH * C1;
    //std::cout << "tH=\n" << tH << "\n";
    H = C2.inverse() * tH;

    //std::cout << "H=\n" << H << "\n";
    H = H / H(2, 2);

    //std::cout << "H=\n" << H << "\n";
    
    return true;
}

void print_mtxf(const Eigen::MatrixXf& X)  
{
  int i, j, nrow, ncol;
   
  nrow = X.rows();
  ncol = X.cols();
  
  lcd.printf("nrow: %d ",nrow);
  lcd.printf("ncol: %d ",ncol);       
  lcd.println();
  
  for (i=0; i<nrow; i++)
  {
    for (j=0; j<ncol; j++)
    {
      lcd.print(X(i,j), 6);   // print 6 decimal places
      lcd.print(", ");
    }
    lcd.println();
  }
  lcd.println();
}

void serialprint_mtxf(const Eigen::MatrixXf& X)  
{
  int i, j, nrow, ncol;
   
  nrow = X.rows();
  ncol = X.cols();
  
  Serial.printf("nrow: %d \n",nrow);
  Serial.printf("ncol: %d \n",ncol);       
  
  for (i=0; i<nrow; i++)
  {
    for (j=0; j<ncol; j++)
    {
      Serial.printf("%f,",X(i,j));   // print 6 decimal places
    }
    Serial.printf("\n");
  }
  Serial.printf("\n");
}
