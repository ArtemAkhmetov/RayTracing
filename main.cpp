#include <iostream>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include "Bitmap.h"
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

template<class T>
void swap(T &a, T &b) {
    T c = b;
    b = a;
    a = c;
}

template<class T>
T max(T a, T b) { return a > b ? a : b; }

template<class T>
T min(T a, T b) { return a > b ? b : a; }

class Vec4f {
public:
    float x, y, z, w;
    Vec4f() : x(0), y(0), z(0), w(0) {} 
    Vec4f(float x_) : x(x_), y(x_), z(x_), w(x_) {} 
    Vec4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {} 
    Vec4f operator*(const float &c) const { return Vec4f(x * c, y * c, z * c, w * c); } 
    Vec4f operator*(const Vec4f &v) const { return Vec4f(x * v.x, y * v.y, z * v.z, w * v.w); } 
    Vec4f operator-(const Vec4f &v) const { return Vec4f(x - v.x, y - v.y, z - v.z, w - v.w); } 
    Vec4f operator+(const Vec4f &v) const { return Vec4f(x + v.x, y + v.y, z + v.z, w + v.w); } 
    Vec4f operator-() const { return Vec4f(-x, -y, -z, -w); } 
    Vec4f& operator+=(const Vec4f &v) { x += v.x, y += v.y, z += v.z, w += v.w; return *this; } 
    friend Vec4f operator*(const float &c, const Vec4f &v) { 
        return Vec4f(v.x * c, v.y * c, v.z * c, v.w * c); 
    } 
    friend std::ostream& operator<<(std::ostream &os, const Vec4f &v) {
        return os << v.x << ", " << v.y << ", " << v.z << ", " << v.w; 
    } 
}; 

class Vec3f {
public:
    float x, y, z;
    Vec3f() : x(0), y(0), z(0) {} 
    Vec3f(float x_) : x(x_), y(x_), z(x_) {} 
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3f(Vec4f v) : x(v.x), y(v.y), z(v.z) {}
    Vec3f operator*(const float &c) const { return Vec3f(x * c, y * c, z * c); } 
    Vec3f operator*(const Vec3f &v) const { return Vec3f(x * v.x, y * v.y, z * v.z); } 
    Vec3f operator-(const Vec3f &v) const { return Vec3f(x - v.x, y - v.y, z - v.z); } 
    Vec3f operator+(const Vec3f &v) const { return Vec3f(x + v.x, y + v.y, z + v.z); } 
    Vec3f operator-() const { return Vec3f(-x, -y, -z); } 
    Vec3f& operator+=(const Vec3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
    Vec3f& operator*=(const Vec3f &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    Vec3f& operator*=(const float &c) { x *= c, y *= c, z *= c; return *this; }
    Vec3f& operator=(const Vec3f &v) { x = v.x, y = v.y, z = v.z; return *this; }
    Vec3f& operator=(const float &c) { x = c, y = c, z = c; return *this; }
    float get_norm() const { return sqrtf(x*x + y*y + z*z); }
    float get_norm2() const { return x*x + y*y + z*z; }
    friend Vec3f operator*(const float &c, const Vec3f &v) { 
        return Vec3f(v.x * c, v.y * c, v.z * c); 
    } 
    friend std::ostream& operator<<(std::ostream &os, const Vec3f &v) {
        return os << v.x << ", " << v.y << ", " << v.z; 
    } 
}; 

class Mat4f {
    float **A;
public:
    Mat4f() {
        A = new float*[4];
        for (int i = 0; i < 4; ++i) {
            A[i] = new float[4]();
        }
    }
    Mat4f(float x) {
        A = new float*[4];
        for (int i = 0; i < 4; ++i) {
            A[i] = new float[4];
            for (int j = 0; j < 4; ++j) {
                if (i == j)
                    A[i][j] = x;
                else
                    A[i][j] = 0.0f;
            }
        }
    }
    Mat4f(Vec3f vec) {
        A = new float*[4];
        for (int i = 0; i < 4; ++i) {
            A[i] = new float[4];
            for (int j = 0; j < 4; ++j) {
                if (i != j)
                    A[i][j] = 0.0f;
                else if (i == 3)
                    A[i][j] = 1.0f;
            }
        }
        A[0][0] = vec.x; A[1][1] = vec.y; A[2][2] = vec.z;
    }
    
    ~Mat4f() {
        for (int i = 0; i < 4; ++i)
            delete[] A[i];
        delete[] A;
    }

    float** get_A() { return A; }

    Vec4f operator*(const Vec4f &vec) const {
        return Vec4f(A[0][0] * vec.x + A[0][1] * vec.y + A[0][2] * vec.z + A[0][3] * vec.w,
                     A[1][0] * vec.x + A[1][1] * vec.y + A[1][2] * vec.z + A[1][3] * vec.w,
                     A[2][0] * vec.x + A[2][1] * vec.y + A[2][2] * vec.z + A[2][3] * vec.w,
                     A[3][0] * vec.x + A[3][1] * vec.y + A[3][2] * vec.z + A[3][3] * vec.w
                    );
    }

    Vec4f operator*(const Vec3f &vec) const {
        return Vec4f(A[0][0] * vec.x + A[0][1] * vec.y + A[0][2] * vec.z + A[0][3],
                     A[1][0] * vec.x + A[1][1] * vec.y + A[1][2] * vec.z + A[1][3],
                     A[2][0] * vec.x + A[2][1] * vec.y + A[2][2] * vec.z + A[2][3],
                     A[3][0] * vec.x + A[3][1] * vec.y + A[3][2] * vec.z + A[3][3]
                    );
    }

    Mat4f operator*(Mat4f &B) const {
        Mat4f C;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    C.get_A()[i][j] = A[i][k] * B.get_A()[k][j];
        return C;
    }
    void translate(Vec3f vec) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (i != j && j != 3)
                    A[i][j] = 0.0f;
                else if (j != 3)
                    A[i][j] = 1.0f;
            }
        }
        A[0][3] = vec.x; A[1][3] = vec.y; A[2][3] = vec.z; A[3][3] = 1;
    }
    void rotate(float theta, Vec3f vec) {
        A[0][0] = cos(theta) + vec.x * vec.x * (1 - cos(theta));
        A[0][1] = vec.x * vec.y * (1 - cos(theta)) - vec.z * sin(theta);
        A[0][2] = vec.x * vec.z * (1 - cos(theta)) + vec.y * sin(theta);
        A[0][3] = 0;
        A[1][0] = vec.y * vec.x * (1 - cos(theta)) + vec.z * sin(theta);
        A[1][1] = cos(theta) + vec.y * vec.y * (1 - cos(theta));
        A[1][2] = vec.y * vec.z * (1 - cos(theta)) - vec.x * sin(theta);
        A[1][3] = 0;
        A[2][0] = vec.z * vec.x * (1 - cos(theta)) - vec.y * sin(theta);
        A[2][1] = vec.z * vec.y * (1 - cos(theta)) + vec.x * sin(theta);
        A[2][2] = cos(theta) + vec.z * vec.z * (1 - cos(theta));
        A[2][3] = 0;
        A[3][0] = 0;
        A[3][1] = 0;
        A[3][2] = 0;
        A[3][3] = 1;
    }
    void transpose() {
        swap<float>(A[0][1], A[1][0]);
        swap<float>(A[0][2], A[2][0]);
        swap<float>(A[0][3], A[3][0]);
        swap<float>(A[1][2], A[2][1]);
        swap<float>(A[1][3], A[3][1]);
        swap<float>(A[2][3], A[3][2]);
    }
    float det() {
        return A[0][0] * (A[1][1]*A[2][2]*A[3][3] + A[2][1]*A[3][2]*A[1][3] + A[1][2]*A[2][3]*A[3][1] -
                A[1][3]*A[2][2]*A[3][1] - A[1][1]*A[2][3]*A[3][2] - A[1][2]*A[2][1]*A[3][3]) -
            A[0][1] * (A[1][0]*A[2][2]*A[3][3] + A[2][0]*A[3][2]*A[1][3] + A[1][2]*A[2][3]*A[3][0] -
                A[1][3]*A[2][2]*A[3][0] - A[1][0]*A[2][3]*A[3][2] - A[2][0]*A[1][2]*A[3][3])  +
            A[0][2] * (A[1][0]*A[2][1]*A[3][3] + A[2][0]*A[3][1]*A[1][3] + A[1][1]*A[2][3]*A[3][0] -
                A[1][3]*A[2][1]*A[3][0] - A[1][0]*A[2][3]*A[3][1] - A[1][1]*A[2][0]*A[3][3]) -
            A[0][3] * (A[1][0]*A[2][1]*A[3][2] + A[2][0]*A[3][1]*A[1][2] + A[1][1]*A[2][2]*A[3][0] -
                A[1][2]*A[2][1]*A[3][0] - A[1][0]*A[3][1]*A[2][2] - A[1][1]*A[2][0]*A[3][2]);
    }
};

Vec3f normalize(const Vec3f &v) { 
    float sqrNorm = v.x * v.x + v.y * v.y + v.z * v.z; 
    if (sqrNorm > 0) { 
        float coef = 1 / sqrtf(sqrNorm); 
        return Vec3f(v.x * coef, v.y * coef, v.z * coef); 
    }
    return v; 
} 
 
inline float dotProduct(const Vec3f &a, const Vec3f &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float radians(const float &deg) { return deg * M_PI / 180; }
inline float abs(const float &a) { return a > 0 ? a : -a; }

Vec3f crossProduct(const Vec3f &a, const Vec3f &b) { 
    return Vec3f( 
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z, 
        a.x * b.y - a.y * b.x 
    ); 
}

Vec3f reflect(const Vec3f &d, const Vec3f &n) { // ||d|| = ||n|| = 1
    return d - 2 * dotProduct(d, n) * n;
}

float norm(const float &x, const float &y, const float &z) { return sqrtf(x*x + y*y + z*z); }

bool planeIntersect(const Vec3f &rayStart, const Vec3f &rayDir, const Vec3f &normal, 
        const float &d, float &t, Vec3f &intersection) {
    float c = dotProduct(rayDir, normal);
    if (c == 0)
        return false;
    t = (d - dotProduct(rayStart, normal)) / c;
    if (t < 0)
        return false;
    intersection = rayStart+t*rayDir;
    return true;
}

bool triangleIntersect(const Vec3f &rayStart, const Vec3f &rayDir, const Vec3f &v0,
        const Vec3f &v1, const Vec3f &v2, float &t, Vec3f &n, Vec3f &intersection) {
    Vec3f e1 = v1 - v0, e2 = v2 - v0;
    n = crossProduct(e1, e2);
    float d = dotProduct(n, v0), a = dotProduct(e1, e1), b = dotProduct(e1, e2),
          c = dotProduct(e2, e2);
    float D = a * c - b * b;
    float A = a / D, B = b / D, C = c / D;
    Vec3f uBeta = C * e1 - B * e2, uGamma = A * e2 - B * e1;
    if (!planeIntersect(rayStart, rayDir, n, d, t, intersection))
        return false;
    Vec3f r = rayStart + t * rayDir - v0;
    float beta = dotProduct(uBeta, r);
    if (beta < 0)
        return false;
    float gamma = dotProduct(uGamma, r);
    if (gamma < 0)
        return false;
    if (1 - beta - gamma < 0)
        return false;
    return true;
}

bool quadIntersect(const Vec3f &rayStart, const Vec3f &rayDir, const Vec3f &v0,
        const Vec3f &v1, const Vec3f &v2, const Vec3f &v3, float &t, Vec3f &n, Vec3f &intersection) {
    if (triangleIntersect(rayStart, rayDir, v0, v1, v2, t, n, intersection))
        return true;
    if (triangleIntersect(rayStart, rayDir, v0, v2, v3, t, n, intersection))
        return true;
    return false;
}

class Light {
public:
    Vec3f pos, color;
    float intensity;
    Light(const Vec3f &pos_, const Vec3f &color_, const float intensity_ = 1) :
        pos(pos_), color(color_), intensity(intensity_) {}
};

class Sphere {
public:
    Vec3f center;
    float radius, radius2;
    float diffuseStrength, shininess, specularStrength;
    Vec3f surfaceColor;
    float transparency, reflection;

    Sphere(const Vec3f &center_ = Vec3f(0.0f, 0.0f, 0.0f), const float &radius_ = 0.5f, 
            const Vec3f &surfaceColor_ = Vec3f(1.0f, 0.0f, 0.0f), 
            const float &diffuseStrength_ = 1.0f, const float &shininess_ = 32, 
            const float &specularStrength_ = 0.5f, const float &reflection_ = 0, const float &transparency_ = 0
            ) : center(center_), radius(radius_), 
            radius2(radius_ * radius_), surfaceColor(surfaceColor_), diffuseStrength(diffuseStrength_),
            shininess(shininess_), specularStrength(specularStrength_),
            transparency(transparency_), reflection(reflection_) {}

    bool intersect(const Vec3f &rayStart, const Vec3f &rayDir, float &t0, float &t1) const {
        Vec3f l = center - rayStart;
        float tca = dotProduct(l, rayDir);
        if (tca < 0) 
            return false;
        float d2 = dotProduct(l, l) - tca * tca;
        if (d2 > radius2) 
            return false;
        float thc = sqrtf(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        return true;
    }
};

class Pyramid {
public:
    Vec3f p1, p2, p3, p4;
    float diffuseStrength, shininess, specularStrength;
    Vec3f surfaceColor;
    float transparency, reflection;

    Pyramid(const Vec3f &p1_ = Vec3f(0.0f, 0.1f, 0.29f), const Vec3f &p2_ = Vec3f(-0.5f, -1.0f, 0.0f),
            const Vec3f &p3_ = Vec3f(0.0f, -1.0f, 0.87f), const Vec3f &p4_ = Vec3f(0.5f, -1.0f, 0.0f),
            const Vec3f &surfaceColor_ = Vec3f(0.0f, 0.0f, 1.0f), 
            const float &diffuseStrength_ = 1.0f, const float &shininess_ = 32, 
            const float &specularStrength_ = 0.5f, const float &reflection_ = 0, const float &transparency_ = 0) : 
            p1(p1_), p2(p2_), p3(p3_), p4(p4_), 
            surfaceColor(surfaceColor_), diffuseStrength(diffuseStrength_),
            shininess(shininess_), specularStrength(specularStrength_), 
            transparency(transparency_), reflection(reflection_) {}

    void translate(const Vec3f &v) {
        p1 += v; p2 += v; p3 += v; p4 += v;
    }

    bool intersect(const Vec3f &rayStart, const Vec3f &rayDir, float &t, Vec3f &intersection, Vec3f &normal) const {
        float a[4] = { 100.0f, 100.0f, 100.0f, 100.0f }, min = 100.0f;
        int imin = 0;
        Vec3f n, p;
        if (triangleIntersect(rayStart, rayDir, p1, p2, p3, t, n, p))
            a[0] = p.get_norm2();
        if (triangleIntersect(rayStart, rayDir, p1, p2, p4, t, n, p))
            a[1] = p.get_norm2();
        if (triangleIntersect(rayStart, rayDir, p1, p3, p4, t, n, p))
            a[2] = p.get_norm2();
        if (triangleIntersect(rayStart, rayDir, p2, p3, p4, t, n, p))
            a[3] = p.get_norm2();
        for (int i = 0; i < 4; ++i) {
            if (a[i] < min) {
                min = a[i];
                imin = i;
            }
        }
        if (a[imin] == 100.0f)
            return false;
        switch (imin) {
            case 0:
                triangleIntersect(rayStart, rayDir, p1, p2, p3, t, normal, intersection);
                break;
            case 1:
                triangleIntersect(rayStart, rayDir, p1, p2, p4, t, normal, intersection);
                break;
            case 2:
                triangleIntersect(rayStart, rayDir, p1, p3, p4, t, normal, intersection);
                break;
            case 3:
                triangleIntersect(rayStart, rayDir, p2, p3, p4, t, normal, intersection);
                break;
        }
        return true;
    }
};

class Cylinder {
public:
    float x, z, y0, y1;                         
    float radius, radius2;              
    float diffuseStrength, shininess, specularStrength;
    Vec3f surfaceColor;   
    float transparency, reflection;

    Cylinder(const float &x_ = 0.0f, const float &z_ = 0.0f, const float &y0_ = -0.5f, 
            const float &y1_ = 0.5f, const float &radius_ = 0.5f, 
            const Vec3f &surfaceColor_ = Vec3f(1.0f, 0.0f, 0.0f), 
            const float &diffuseStrength_ = 1.0f, const float &shininess_ = 32, 
            const float &specularStrength_ = 0.5f, const float &reflection_ = 0, const float &transparency_ = 0
            ) : x(x_), z(z_), y0(y0_), y1(y1_), radius(radius_), 
            radius2(radius_ * radius_), surfaceColor(surfaceColor_), diffuseStrength(diffuseStrength_),
            shininess(shininess_), specularStrength(specularStrength_), 
            transparency(transparency_), reflection(reflection_) {}

    bool intersect(const Vec3f &rayStart, const Vec3f &rayDir, float &t0, float &t1, Vec3f &intersection, Vec3f &normal) const {
        Vec3f l = Vec3f(x - rayStart.x, 0.0f, z - rayStart.z);
        Vec3f k = normalize(Vec3f(rayDir.x, 0.0f, rayDir.z));
        float tca = dotProduct(l, k);
        if (tca < 0) 
            return false;
        float d2 = dotProduct(l, l) - tca * tca;
        if (d2 > radius2) 
            return false;
        float thc = sqrtf(radius2 - d2);
        t0 = (tca - thc) / dotProduct(k, rayDir);
        intersection = rayStart + t0 * rayDir;
        normal = normalize(Vec3f(intersection.x - x, 0.0f, intersection.z - z));
        if (intersection.y > y1) {
            if (planeIntersect(rayStart, rayDir, Vec3f(0.0f, 1.0f, 0.0f), y1, t0, intersection)) {
                if ((intersection.x - x) * (intersection.x - x) + (intersection.z - z) * (intersection.z - z) > radius2)
                    return false;
                normal = Vec3f(0.0f, 1.0f, 0.0f);
            }
            else
                return false;
        }
        if (intersection.y < y0) {
            if (planeIntersect(rayStart, rayDir, Vec3f(0.0f, 1.0f, 0.0f), y0, t0, intersection)) {
                if ((intersection.x - x) * (intersection.x - x) + (intersection.z - z) * (intersection.z - z) > radius2)
                    return false;
                normal = Vec3f(0.0f, -1.0f, 0.0f);
            }
            else
                return false;
        }
        t1 = (tca + thc) / dotProduct(k, rayDir);
        return true;
    }
};

const int imageWidth = 1024;
const int imageHeight = 1024;

uint32_t colorNum(Vec3f color) {
    uint32_t red=(uint32_t)0x000000FF * (color.x > 1 ? 1 : color.x), 
             green=(uint32_t)0x000000FF * (color.y > 1 ? 1 : color.y),
             blue=(uint32_t)0x000000FF * (color.z > 1 ? 1 : color.z);
    green <<= 8;
    blue <<= 16;
    return red+green+blue; 
}

class Room {
public:
    Vec3f floorColor, upColor, rightColor, leftColor, backColor;
    Room(Vec3f floorColor_=0, Vec3f upColor_=0, Vec3f rightColor_=0, Vec3f leftColor_=0, Vec3f backColor_=0) : 
    floorColor(floorColor_), upColor(upColor_), rightColor(rightColor_), leftColor(leftColor_), backColor(backColor_) {}
};

bool intersect_up(const Vec3f &rayStart, const float &level, const Vec3f &rayDir, float &x, float &z) {
    x = rayStart.x + ((level-rayStart.y)/(dotProduct(rayDir, Vec3f(0.0f, 1.0f, 0.0f)))) * rayDir.x;
    z = rayStart.z + ((level-rayStart.y)/(dotProduct(rayDir, Vec3f(0.0f, 1.0f, 0.0f)))) * rayDir.z;
    if (-1.0f <= x && x <= 1.0f && -1.0f <= z && z <= 1.0f) 
        return true;
    return false;
}

bool intersect_floor(const Vec3f &rayStart, const float &level, const Vec3f &rayDir, float &x, float &z) {
    x = rayStart.x + ((rayStart.y - level)/(dotProduct(rayDir, Vec3f(0.0f, -1.0f, 0.0f)))) * rayDir.x;
    z = rayStart.z + ((rayStart.y - level)/(dotProduct(rayDir, Vec3f(0.0f, -1.0f, 0.0f)))) * rayDir.z;
    if (-1.0f <= x && x <= 1.0f && -1.0f <= z && z <= 1.0f) 
        return true;
    return false;
}

bool intersect_right(const Vec3f &rayStart, const float &level, const Vec3f &rayDir, float &y, float &z) {
    y = rayStart.y + ((level-rayStart.x)/(dotProduct(rayDir, Vec3f(1.0f, 0.0f, 0.0f)))) * rayDir.y;
    z = rayStart.z + ((level-rayStart.x)/(dotProduct(rayDir, Vec3f(1.0f, 0.0f, 0.0f)))) * rayDir.z;
    if (-1.0f <= y && y <= 1.0f && -1.0f <= z && z <= 1.0f) 
        return true;
    return false;
}

bool intersect_left(const Vec3f &rayStart, const float &level, const Vec3f &rayDir, float &y, float &z) {
    y = rayStart.y + ((rayStart.x - level)/(dotProduct(rayDir, Vec3f(-1.0f, 0.0f, 0.0f)))) * rayDir.y;
    z = rayStart.z + ((rayStart.x - level)/(dotProduct(rayDir, Vec3f(-1.0f, 0.0f, 0.0f)))) * rayDir.z;
    if (-1.0f <= y && y <= 1.0f && -1.0f <= z && z <= 1.0f) 
        return true;
    return false;
}

bool intersect_back(const Vec3f &rayStart, const float &level, const Vec3f &rayDir, float &x, float &y) {
    x = rayStart.x + ((rayStart.z - level)/(dotProduct(rayDir, Vec3f(0.0f, 0.0f, -1.0f)))) * rayDir.x;
    y = rayStart.y + ((rayStart.z - level)/(dotProduct(rayDir, Vec3f(0.0f, 0.0f, -1.0f)))) * rayDir.y;
    if (-1.0f <= x && x <= 1.0f && -1.0f <= y && y <= 1.0f) 
        return true;
    return false;
}

#define MAX_RAY_DEPTH 4

Vec3f trace(const Vec3f &rayStart, const Vec3f &rayDir, const std::vector<Light> &lights, 
        const std::vector<Sphere> &spheres, const Room &room, const int &depth) {
    float tnear = INFINITY, t0 = INFINITY, t1 = INFINITY, specularStrength=0.5f, x0, y0, z0, shadow=1, shininess=32, spec, diff, bias = 1e-4;
    Vec3f intersection, result, diffuse, norm, specular, reflection, refraction, lightDir, reflectDir, viewDir, shadowDir, shadowVector;
    int index = -1, transmission;

    for (int i = 0; i < spheres.size(); ++i) {
        if (spheres[i].intersect(rayStart, rayDir, t0, t1) && t0 < tnear) {
            tnear = t0;
            index = i;
        }
    }
    
    if (index == -1) {
        if (intersect_floor(rayStart, -1.0f, rayDir, x0, z0)) {//floor
            for (int j = 0; j < lights.size(); ++j) {
                norm = Vec3f(0.0f, 1.0f, 0.0f);
                lightDir = normalize(lights[j].pos - Vec3f(x0, -1.0f, z0));
                diffuse = max<float>(dotProduct(norm, lightDir), 0.0f) * lights[j].color;
                spec = pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, norm)), 0.0f), shininess);
                specular = spec * lights[j].color;
                shadowVector = Vec3f(lights[j].pos.x-x0, lights[j].pos.y+1.0f, lights[j].pos.z-z0);
                shadowDir = normalize(shadowVector);
                for (int i = 0; i < spheres.size(); ++i) {
                    if (spheres[i].intersect(Vec3f(x0, -1.0f, z0), shadowDir, t0, t1)) {
                        if ((t0*shadowDir).get_norm2() < shadowVector.get_norm2()) {
                            shadow *= 0.2f;
                            break;
                        }
                    }
                }
                if (shadow == 1.0f)
                    result += (diffuse + specularStrength * specular) * room.floorColor;
                else
                    result = room.floorColor;
            }
        }
        else if (intersect_up(rayStart, 1.0f, rayDir, x0, z0)) {//up
            for (int j = 0; j < lights.size(); ++j) {
                norm = Vec3f(0.0f, -1.0f, 0.0f);
                lightDir = normalize(lights[j].pos - Vec3f(x0, 1.0f, z0));
                diff = max<float>(dotProduct(norm, lightDir), 0.0f);
                diffuse = diff * lights[j].color;
                viewDir = -rayDir;
                reflectDir = reflect(-lightDir, norm);
                spec = pow(max<float>(dotProduct(viewDir, reflectDir), 0.0f), shininess);
                specular = specularStrength * spec * lights[j].color;
                shadowVector = Vec3f(x0-lights[j].pos.x, 1.0f-lights[j].pos.y, z0-lights[j].pos.z);
                shadowDir = normalize(shadowVector);
                for (int i = 0; i < spheres.size(); ++i) {
                    if (spheres[i].intersect(lights[j].pos, shadowDir, t0, t1)) {
                        if ((t0*shadowDir).get_norm2() < shadowVector.get_norm2()) {
                            shadow *= 0.2f;
                            break;
                        }
                    }
                }
                if (shadow == 1.0f)
                    result += (diffuse + specular) * room.upColor;
                else
                    result = room.upColor;
            }
        }
        else if (intersect_right(rayStart, 1.0f, rayDir, y0, z0)) {//right
            for (int j = 0; j < lights.size(); ++j) {
                norm = Vec3f(-1.0f, 0.0f, 0.0f);
                lightDir = normalize(lights[j].pos - Vec3f(1.0f, y0, z0));
                diff = max<float>(dotProduct(norm, lightDir), 0.0f);
                diffuse = diff * lights[j].color;
                viewDir = -rayDir;
                reflectDir = reflect(-lightDir, norm);
                spec = pow(max<float>(dotProduct(viewDir, reflectDir), 0.0f), shininess);
                specular = specularStrength * spec * lights[j].color;
                shadowVector = Vec3f(lights[j].pos.x-1.0f, lights[j].pos.y-y0, lights[j].pos.z-z0);
                shadowDir = normalize(shadowVector);
                for (int i = 0; i < spheres.size(); ++i) {
                    if (spheres[i].intersect(Vec3f(1.0f, y0, z0), shadowDir, t0, t1)) {
                        if ((t0*shadowDir).get_norm2() < shadowVector.get_norm2()) {
                            shadow *= 0.2f;
                            break;
                        }
                    }
                }
                if (shadow == 1.0f)
                    result += (diffuse + specular) * room.rightColor;
                else
                    result = room.rightColor;
            }
        }
        else if (intersect_left(rayStart, -1.0f, rayDir, y0, z0)) {//left
            for (int j = 0; j < lights.size(); ++j) {
                norm = Vec3f(1.0f, 0.0f, 0.0f);
                lightDir = normalize(lights[j].pos - Vec3f(-1.0f, y0, z0));
                diff = max<float>(dotProduct(norm, lightDir), 0.0f);
                diffuse = diff * lights[j].color;
                viewDir = -rayDir;
                reflectDir = reflect(-lightDir, norm);
                spec = pow(max<float>(dotProduct(viewDir, reflectDir), 0.0f), shininess);
                specular = specularStrength * spec * lights[j].color;
                shadowVector = Vec3f(lights[j].pos.x+1.0f, lights[j].pos.y-y0, lights[j].pos.z-z0);
                shadowDir = normalize(shadowVector);
                for (int i = 0; i < spheres.size(); ++i) {
                    if (spheres[i].intersect(Vec3f(-1.0f, y0, z0), shadowDir, t0, t1)) {
                        if ((t0*shadowDir).get_norm2() < shadowVector.get_norm2()) {
                            shadow *= 0.2f;
                            break;
                        }
                    }
                }
                if (shadow == 1.0f)
                    result += (diffuse + specular) * room.leftColor;
                else
                    result = room.leftColor;
            }
        }
        else if (intersect_back(rayStart, -1.0f, rayDir, x0, y0)) {//back
            for (int j = 0; j < lights.size(); ++j) {
                norm = Vec3f(0.0f, 0.0f, 1.0f);
                lightDir = normalize(lights[j].pos - Vec3f(x0, y0, -1.0f));
                diff = max<float>(dotProduct(norm, lightDir), 0.0f);
                diffuse = diff * lights[j].color;
                viewDir = -rayDir;
                reflectDir = reflect(-lightDir, norm);
                spec = pow(max<float>(dotProduct(viewDir, reflectDir), 0.0f), shininess);
                specular = specularStrength * spec * lights[j].color;
                shadowVector = Vec3f(lights[j].pos.x-x0, lights[j].pos.y-y0, lights[j].pos.z+1.0f);
                shadowDir = normalize(shadowVector);
                for (int i = 0; i < spheres.size(); ++i) {
                    if (spheres[i].intersect(Vec3f(x0, y0, -1.0f), shadowDir, t0, t1)) {
                        if ((t0*shadowDir).get_norm2() < shadowVector.get_norm2()) {
                            shadow *= 0.2f;
                            break;
                        }
                    }
                }
                if (shadow == 1.0f)
                    result += (diffuse + specular) * room.backColor;
                else
                    result = room.backColor;
            }
        }
        else 
            result = Vec3f(1.0f, 1.0f, 1.0f);
        return result * shadow;
    }

    intersection = rayStart + tnear * rayDir;
    norm = normalize(intersection - spheres[index].center);
    diffuse = Vec3f(0.0f, 0.0f, 0.0f);
    specular = Vec3f(0.0f, 0.0f, 0.0f);
    reflection = Vec3f(0.0f, 0.0f, 0.0f);
    refraction = Vec3f(0.0f, 0.0f, 0.0f);
    result = Vec3f(0.0f, 0.0f, 0.0f);

    if (depth < MAX_RAY_DEPTH) {
        for (int i = 0; i < lights.size(); ++i) {
            transmission = 1;
            lightDir = normalize(lights[i].pos - intersection);
            for (int j = 0; j < spheres.size(); ++j) {
                if (spheres[j].intersect(intersection + norm * bias, lightDir, t0, t1)) {
                    transmission = 0;
                    break;
                }
            }
            diffuse += transmission * max<float>(0.0f, dotProduct(norm, lightDir)) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, norm)), 0.0f), spheres[index].shininess) * lights[i].color;
        }
        if (spheres[index].reflection > 0) {
            reflection = trace(intersection + norm * bias, reflect(rayDir, norm), lights, spheres, room, depth + 1);
        }
        if (spheres[index].transparency > 0) {
            float eta = 0.9f;
            float cosi = -dotProduct(norm, rayDir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = normalize(rayDir * eta + norm * (eta *  cosi - sqrtf(k)));
            refraction = trace(intersection - norm * bias, refrdir, lights, spheres, room, depth + 1);
        }
        result = spheres[index].surfaceColor * diffuse * spheres[index].diffuseStrength + 
            spheres[index].specularStrength * specular + 
            (reflection * spheres[index].reflection + refraction);
    }
    return result;
}

void renderSphere(std::vector<uint32_t> &image, const int &width, const int &height, const std::vector<Sphere> &spheres, 
        const Vec3f &cameraPos, const std::vector<Light> &lights, const Room &room) {
    #pragma omp parallel for
    for (int x1 = 0; x1 < width; ++x1) {
        for (int y1 = 0; y1 < height; ++y1) {
            image[y1*width+x1] = colorNum(trace(Vec3f((float)2 * x1 / width - 1, (float)2 * y1 / height - 1, 1), 
                normalize(Vec3f((float)2 * x1 / width - 1 - cameraPos.x, (float)2 * y1 / height - 1 - cameraPos.y, 1 - cameraPos.z)), 
                lights, spheres, room, 1));
        }
    }
}

bool isDiag(const float &x, const float &z, const int &size) {
    int n;
    n = (int)(size*(x+1)) + (int)(size*(z+1));
    if (n >= 1 && n <= 4*size-3 && n % 2 == 1)
        return true;
    return false;
}

Vec3f traceCylinder(const Vec3f &rayStart, const Vec3f &rayDir, const std::vector<Light> &lights, 
        const std::vector<Cylinder> &cylinders, const int &depth) {
    float x, y, t, t0=INFINITY, t1=INFINITY, tnear=INFINITY, specularStrength=0.5f, shininess=32, bias = 1e-4;
    Vec3f normal, diffuse, specular, result, lightDir, intersection, planeColor, n, reflectColor;
    int index = -1, transmission;
    if (depth > MAX_RAY_DEPTH)
        return Vec3f(0.0f, 0.0f, 0.0f);
    diffuse = Vec3f(0.0f, 0.0f, 0.0f);
    specular = Vec3f(0.0f, 0.0f, 0.0f);
    result = Vec3f(1.0f, 1.0f, 1.0f);
    planeColor = Vec3f(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < cylinders.size(); ++i) {
        if (cylinders[i].intersect(rayStart, rayDir, t0, t1, intersection, normal) && t0 < tnear) {
            tnear = t0;
            index = i;
        }
    }
    if (index != -1 && cylinders[index].intersect(rayStart, rayDir, t0, t1, intersection, normal)) {
        for (int i = 0; i < lights.size(); ++i) {
            lightDir = normalize(lights[i].pos - intersection);
            transmission = 1;
            for (int j = 0; j < cylinders.size(); ++j) {
                if (cylinders[j].intersect(intersection + normal * bias, lightDir, t0, t1, n, n)) {
                    transmission = 0;
                    break;
                }
            }
            diffuse += transmission * max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), cylinders[index].shininess) * 
                lights[i].color;
            if (cylinders[index].reflection > 0)
                reflectColor = traceCylinder(intersection + bias * normal, reflect(rayDir, normal), lights, cylinders, depth+1);
        }
        result = (diffuse + cylinders[index].specularStrength * specular) * cylinders[index].surfaceColor + 
            cylinders[index].reflection * reflectColor;
    }
    else if (planeIntersect(rayStart, rayDir, Vec3f(0.0f, 1.0f, 0.0f), -1.0f, t, intersection) &&
            intersection.x < 1 &&
            intersection.x > -1 &&
            intersection.z < 1 &&
            intersection.z > -1) {
        if (isDiag(intersection.x, intersection.z, 3))
            planeColor = Vec3f(0.07f, 0.07f, 0.07f);
        for (int i = 0; i < lights.size(); ++i) {
            transmission = 1;
            for (int j = 0; j < cylinders.size(); ++j) {
                if (cylinders[j].intersect(intersection, normalize(lights[i].pos - intersection), 
                            t0, t1, n, n)) {
                    transmission = 0;
                    break;
                }
            }
            normal = Vec3f(0.0f, 1.0f, 0.0f);
            lightDir = normalize(lights[i].pos - intersection);
            diffuse += transmission * max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), shininess) * 
                lights[i].color;
        }
        result = (diffuse + specular * specularStrength) * planeColor;
    }
    else if (planeIntersect(rayStart, rayDir, Vec3f(-1.0f, 0.0f, 0.0f), -1.0f, t, intersection) &&
            intersection.y < 0.1 &&
            intersection.y > -1 &&
            intersection.z < 0.95 &&
            intersection.z > -0.2) {
        if ((intersection.y + 0.5f) * (intersection.y + 0.5f) + (intersection.z - 0.4) * (intersection.z - 0.4) <= 0.25f) {
            reflectColor = traceCylinder(intersection + bias * Vec3f(-1.0f, 0.0f, 0.0f), 
                reflect(rayDir, Vec3f(-1.0f, 0.0f, 0.0f)), lights, cylinders, depth+1);
            result = reflectColor;
        }
        else {
            for (int i = 0; i < lights.size(); ++i) {
                transmission = 1;
                for (int j = 0; j < cylinders.size(); ++j) {
                    if (cylinders[j].intersect(intersection, normalize(lights[i].pos - intersection), 
                                t0, t1, n, n)) {
                        transmission = 0;
                        break;
                    }
                }
                normal = Vec3f(-1.0f, 0.0f, 0.0f);
                lightDir = normalize(lights[i].pos - intersection);
                diffuse += transmission * max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
                specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), shininess) * 
                    lights[i].color;
            }
            result = (diffuse + specular * specularStrength) * Vec3f(0.3,0.3,0.3);
        }
    }
    else
        result = Vec3f(0.0f, 0.0f, 0.0f);
    return result;
}

void renderCylinder(std::vector<uint32_t> &image, const int &width, const int &height, const std::vector<Cylinder> &cylinders, 
        const Vec3f &cameraPos, const std::vector<Light> &lights) {
    #pragma omp parallel for
    for (int x1 = 0; x1 < width; ++x1) {
        for (int y1 = 0; y1 < height; ++y1) {
            image[y1*width+x1] = colorNum(traceCylinder(Vec3f((float)2 * x1 / width - 1, (float)2 * y1 / height - 1, 1), 
                normalize(Vec3f((float)2 * x1 / width - 1 - cameraPos.x, (float)2 * y1 / height - 1 - cameraPos.y, 1 - cameraPos.z)), 
                lights, cylinders, 1));
        }
    }
}

bool cubeIntersect(const Vec3f &rayStart, const Vec3f &rayDir, Vec3f *n, float *d, float &alpha, Vec3f &normal) {
    float fMax = -99999, bMin = 99999, s;
    int imax, imin;
    for (int i = 0; i < 6; ++i) {
        s = dotProduct(rayDir, n[i]);
        if (s == 0) {
            if (dotProduct(rayStart, n[i]) > d[i])
                return false;
            else
                continue;
        }
        alpha = (d[i] - dotProduct(rayStart, n[i])) / s;
        if (dotProduct(rayDir, n[i]) < 0) {
            if (alpha > fMax) {
                if (alpha > bMin)
                    return false;
                fMax = alpha;
                imax = i;
            }
        }
        else {
            if (alpha < bMin) {
                if (alpha < 0 || alpha < fMax)
                    return false;
                bMin = alpha;
                imin = i;
            }
        }
    }
    normal = fMax > 0 ? n[imax] : n[imin];
    alpha = fMax > 0 ? fMax : bMin;
    return true;
}

bool isDiagInf(const float &x, const float &z) {
    int n;
    n = (int)(4*(x+50)) + (int)(4*(z+510));
    if (n >= 1 && n <= 4400 && n % 2 == 1)
        return true;
    return false;
}

Vec3f scene1trace(const Vec3f &rayStart, const Vec3f &rayDir, const Vec3f &objectColor, 
        Vec3f *n, float *d, const std::vector<Light> &lights, const Vec3f &backgroundColor, const int &depth) {
    float t, t0, t1, alpha, shininess=32, specularStrength=0.5f, bias=1e-4;
    int transmission;
    Vec3f normal, diffuse, specular, result, lightDir, intersection, planeColor, reflection, refraction, p;
    diffuse = Vec3f(0.0f, 0.0f, 0.0f);
    specular = Vec3f(0.0f, 0.0f, 0.0f);
    result = Vec3f(1.0f, 1.0f, 1.0f);
    planeColor = Vec3f(1.0f, 1.0f, 1.0f);
    Sphere sphere(Vec3f(0.3f, -0.3f, 0.1f), 0.6f, Vec3f(0.97f, 0.97f, 0.97f), 0, 1425, 10.0f, 0.8f, 0);
    Pyramid pyramid;
    pyramid.translate(Vec3f(-1.3f, 0.0f, -2.75f));
    if (depth > MAX_RAY_DEPTH)
        return Vec3f(0.0f, 0.0f, 0.0f);
    if (cubeIntersect(rayStart, rayDir, n, d, alpha, normal)) {
        intersection = rayStart + alpha * rayDir;
        for (int i = 0; i < lights.size(); ++i) {
            transmission = 1;
            if (sphere.intersect(intersection, normalize(lights[i].pos - intersection), t0, t1))
                transmission = 0;
            lightDir = normalize(lights[i].pos - intersection);
            diffuse += transmission * max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), shininess) * lights[i].color;
        }
        result = (diffuse + specularStrength * specular) * objectColor;
    }
    else if (sphere.intersect(rayStart, rayDir, t0, t1)) {
        intersection = rayStart + t0 * rayDir;
        normal = normalize(intersection - sphere.center);
        for (int i = 0; i < lights.size(); ++i) {
            lightDir = normalize(lights[i].pos - intersection);
            diffuse += max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), sphere.shininess) * lights[i].color;
        }
        if (sphere.reflection > 0) {
            reflection = scene1trace(intersection + normal * bias, reflect(rayDir, normal), objectColor, n, d, lights, backgroundColor, depth + 1);
        }
        if (sphere.transparency > 0) {
            float eta = 0.9f;
            float cosi = -dotProduct(normal, rayDir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = normalize(rayDir * eta + normal * (eta *  cosi - sqrtf(k)));
            refraction = scene1trace(intersection - normal * bias, refrdir, objectColor, n, d, lights, backgroundColor, depth + 1);
        }
        result = sphere.surfaceColor * diffuse * sphere.diffuseStrength + 
            sphere.specularStrength * specular + 
            (reflection * sphere.reflection + refraction);
    }
    else if (pyramid.intersect(rayStart, rayDir, t, intersection, normal)) {
        for (int i = 0; i < lights.size(); ++i) {
            lightDir = normalize(lights[i].pos - intersection);
            diffuse += max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), pyramid.shininess) * lights[i].color;
        }
        if (pyramid.reflection > 0) {
            reflection = scene1trace(intersection + normal * bias, reflect(rayDir, normal), objectColor, n, d, lights, backgroundColor, depth + 1);
        }
        if (pyramid.transparency > 0) {
            float eta = 0.9f;
            float cosi = -dotProduct(normal, rayDir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = normalize(rayDir * eta + normal * (eta *  cosi - sqrtf(k)));
            refraction = scene1trace(intersection - normal * bias, refrdir, objectColor, n, d, lights, backgroundColor, depth + 1);
        }
        result = pyramid.surfaceColor * diffuse * pyramid.diffuseStrength + 
            pyramid.specularStrength * specular + 
            (reflection * pyramid.reflection + refraction);
    }
    else if (planeIntersect(rayStart, rayDir, Vec3f(0.0f, 1.0f, 0.0f), -1.0f, t, intersection)) {
        if (isDiagInf(intersection.x, intersection.z))
            planeColor = Vec3f(0.07f, 0.07f, 0.07f);
        for (int i = 0; i < lights.size(); ++i) {
            transmission = 1;
            if (cubeIntersect(intersection, normalize(lights[i].pos - intersection), n, d, alpha, normal) ||
                sphere.intersect(intersection, normalize(lights[i].pos - intersection), t0, t1) || 
                pyramid.intersect(intersection, normalize(lights[i].pos - intersection), t, p, normal)) {
                transmission = 0;
            }
            normal = Vec3f(0.0f, 1.0f, 0.0f);
            lightDir = normalize(lights[i].pos - intersection);
            diffuse += transmission * max<float>(dotProduct(normal, lightDir), 0.0f) * lights[i].color;
            specular += pow(max<float>(dotProduct(-rayDir, reflect(-lightDir, normal)), 0.0f), shininess) * lights[i].color;
        }
        result = (diffuse + specularStrength * specular) * planeColor;
    }
    else
        result = backgroundColor;
    return result;
}

void renderCube(std::vector<uint32_t> &image, const std::vector<Vec3f> &background, const int &width, const int &height, const Vec3f &objectColor, 
        Vec3f *n, float *d, const Vec3f &cameraPos, const std::vector<Light> &lights) {
    #pragma omp parallel for
    for (int x1 = 0; x1 < width; ++x1) {
        for (int y1 = 0; y1 < height; ++y1) {
            image[y1*width+x1] = colorNum(scene1trace(Vec3f((float)2 * x1 / width - 1, (float)2 * y1 / height - 1, 1), 
                normalize(Vec3f((float)2 * x1 / width - 1 - cameraPos.x, (float)2 * y1 / height - 1 - cameraPos.y, 1 - cameraPos.z)), 
                objectColor, n, d, lights, background[(height - y1 - 1)*width+x1], 1));
        }
    }
}

void rotateVertices(float *vertices, int length, float theta, Vec3f vec) {
    Mat4f mat;
    mat.rotate(theta, vec);
    Vec4f buf;
    #pragma omp parallel for
    for (int i = 0; i < length; i += 3) {
        #pragma omp critical (section2)
        {
            buf.x = vertices[i];
            buf.y = vertices[i+1];
            buf.z = vertices[i+2];
            buf.w = 1;
            buf = mat * buf;
            vertices[i] = buf.x;
            vertices[i+1] = buf.y;
            vertices[i+2] = buf.z;
        }
    }
}

void rotateNormals(Vec3f *normals, int length, float theta, Vec3f vec) {
    Mat4f mat;
    mat.rotate(theta, vec);
    #pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        normals[i] = Vec3f(mat * normals[i]);
    }
}

int main(int argc, char **argv) {
    std::unordered_map<std::string, std::string> cmdLineParams;
    for (int i = 0; i < argc; ++i) {
        std::string key(argv[i]);
        if (key.size() > 0 && key[0] == '-') {
            if (i != argc-1) {
                cmdLineParams[key] = argv[i+1];
                ++i;
            }
            else
                cmdLineParams[key] = "";
        }
    }

    Vec3f cameraPos(0.0f, 0.0f, 2.0f);
    std::vector<Light> lights;
    lights.push_back(Light(Vec3f(0.6f, 0.95f, 0.97f), Vec3f(1.0f, 1.0f, 1.0f)));
    lights.push_back(Light(Vec3f(-0.8f, 0.65f, 0.98f), Vec3f(1.0f, 1.0f, 1.0f)));

    std::vector<uint32_t> image(imageWidth * imageHeight);
    
    int buf = -1, background_width, background_height;
    std::vector<Vec3f> background;
    unsigned char *pixmap = stbi_load("../space.jpg", &background_width, &background_height, &buf, 0);
    if (!pixmap || buf != 3) {
        std::cerr << "Error: can't load the environment map" << std::endl;
        return -1;
    }
    background = std::vector<Vec3f>(background_width * background_height);
    #pragma omp parallel for
    for (int j = background_height-1; j >= 0 ; --j) {
        for (int i = 0; i < background_width; ++i) {
            background[i + j * background_width] = Vec3f(pixmap[(i + j * background_width) * 3], 
                    pixmap[(i + j * background_width) * 3 + 1], pixmap[(i + j * background_width) * 3 + 2])*(1/255.0f);
        }
    }
    stbi_image_free(pixmap);

    for (auto &pixel: image)
        pixel = colorNum(Vec3f(1.0f, 1.0f, 1.0f));

    float cube_vertices[] = {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f, 
        -0.5f,  0.5f, -0.5f,

        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f
    };

    Vec3f normals[] = {
        Vec3f(0,0,-1),
        Vec3f(0,0,1),
        Vec3f(-1,0,0),
        Vec3f(1,0,0),
        Vec3f(0,-1,0),
        Vec3f(0,1,0)
    };

    float d[] = {
        -0.1f, 0.6f, 0.9f, -0.4f, 1.0f, -0.5f
    };

    std::string outFilePath = "zout.bmp";
    if (cmdLineParams.find("-out") != cmdLineParams.end())
        outFilePath = cmdLineParams["-out"];

    int sceneId = 0;
    if (cmdLineParams.find("-scene") != cmdLineParams.end())
        sceneId = atoi(cmdLineParams["-scene"].c_str());

    int n_jobs = 1;
    if (cmdLineParams.find("-threads") != cmdLineParams.end())
        n_jobs = atoi(cmdLineParams["-threads"].c_str());
    omp_set_num_threads(n_jobs);

    if (sceneId == 1) {
        lights.push_back(Light(Vec3f(-0.2f, 0.85f, -0.68f), Vec3f(1.0f, 1.0f, 1.0f)));
        renderCube(image, background, imageWidth, imageHeight, Vec3f(0.9f, 0.1f, 0.1f), normals, d, cameraPos, lights);
    }
    else if (sceneId == 2) {
        std::vector<Cylinder> cylinders;
        cylinders.push_back(Cylinder(0.0f, -0.1f, -1.0f, -0.25f, 0.52f, Vec3f(1.0f, 0.2f, 0.21f), 0.9f, 64, 0.1f, 0.1, 0));
        cylinders.push_back(Cylinder(-0.35f, 0.7f, -1.0f, -0.55f, 0.25f, Vec3f(0.2f, 0.8f, 0.3f), 0.9f, 64, 0.1f, 0, 0));
        renderCylinder(image, imageWidth, imageHeight, cylinders, cameraPos, lights);
    }
    else if (sceneId == 3) {
        std::vector<Sphere> spheres;
        Room room(Vec3f(0.81f, 0.06f, 0.13f),
                    Vec3f(0.3f, 0.56f, 0.67f), Vec3f(0.41f, 0.42f, 0.37f), Vec3f(0.41f, 0.42f, 0.37f), 
                    Vec3f(0.31f, 0.36f, 0.53f));

        spheres.push_back(Sphere(Vec3f(-0.33f, -0.15f, -0.28f), 0.48f, Vec3f(1.0f, 0.5f, 0.31f), 0.9f, 32, 0.1f, 0, 0));
        spheres.push_back(Sphere(Vec3f(0.55f, -0.45f, 0.54f), 0.45f, Vec3f(0.1f, 0.2f, 1.0f), 0.9f, 64, 0.1f, 0, 0));
        spheres.push_back(Sphere(Vec3f(-0.5f, -0.64f, 0.5f), 0.35f, Vec3f(0.1f, 0.97f, 0.2f), 0.9f, 32, 0.1f, 0.08f, 0));
        spheres.push_back(Sphere(Vec3f(0.25f, 0.15f, 0.6f), 0.25f, Vec3f(0.4f, 0.4f, 0.3f), 0.6f, 50, 0.3f, 0.1f, 1));
        renderSphere(image, imageWidth, imageHeight, spheres, cameraPos, lights, room);
    }

    SaveBMP(outFilePath.c_str(), image.data(), imageWidth, imageHeight);

    std::cout << "end." << std::endl;
    return 0;
}
