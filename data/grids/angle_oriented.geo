cl__1 = 1;
Mesh.Optimize = 1;
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 0.1, 0, 0.1};
Point(4) = {0.1, 0.1, 0, 0.1};
Point(5) = {0.1, 1, 0, 0.1};
Point(6) = {0, 1, 0, 0.1};

Point(7) = {0, 0, 0.1, 0.1};
Point(8) = {1, 0, 0.1, 0.1};
Point(9) = {1, 0.1, 0.1, 0.1};
Point(10) = {0.1, 0.1, 0.1,0.1};
Point(11) = {0.1, 1, 0.1,0.1};
Point(12) = {0, 1, 0.1, 0.1};

Rotate {{0,0,1}, {0,0,0}, Pi/4} { Point{1,2,3,4,5,6,7,8,9,10,11,12}; };
Rotate {{1,0,0}, {0,0,0}, Pi/2} { Point{1,2,3,4,5,6,7,8,9,10,11,12}; };


Line(1) = {8, 7};
Line(2) = {7, 1};
Line(3) = {1, 2};
Line(4) = {2, 3};
Line(5) = {3, 9};
Line(6) = {9, 8};
Line(7) = {8, 2};
Line(8) = {12, 6};
Line(9) = {6, 5};
Line(10) = {5, 11};
Line(11) = {11, 12};
Line(12) = {12, 7};
Line(13) = {6, 1};
Line(14) = {5, 4};
Line(15) = {11, 10};
Line(16) = {10, 9};
Line(17) = {3, 4};
Line(18) = {4, 10};
Line Loop(19) = {4,7,6,5};
Plane Surface(20) = {19};
Line Loop(21) = {16, -5, 17, 18};
Plane Surface(22) = -{21};
Line Loop(23) = {15, -18, -14, 10};
Plane Surface(24) = {23};
Line Loop(25) = {10, 11, 8, 9};
Plane Surface(26) = -{25};
Line Loop(27) = {12, 2, -13, -8};
Plane Surface(28) = -{27};
Line Loop(29) = {9, 14, -17, -4, -3, -13};
Plane Surface(30) = -{29};
Line Loop(31) = {6, 1, -12, -11, 15, 16};
Plane Surface(32) = -{31};
Line Loop(33) = {7, -3, -2, -1};
Plane Surface(34) = {33};
Physical Surface(39) = { 28,-30, -34, 20,-22, 24, 26, 32};

