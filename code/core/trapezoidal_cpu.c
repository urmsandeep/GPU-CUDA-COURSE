#include <stdio.h>
#include <math.h>

// Function to integrate
double f(double x) {
    return x * x;  // f(x) = x²
}

double trap_serial(double a, double b, 
                   int n) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;
    
    // Sum middle trapezoids
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += f(x);
    }
    
    return h * sum;
}

int main() {
    double a = 0.0, b = 10.0;
    int n = 1000000;
    
    double result = trap_serial(a, b, n);
    
    printf("Integral = %.6f\n", result);
    // Expected: 333.333333
    return 0;
}
