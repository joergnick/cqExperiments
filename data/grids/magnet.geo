cl__1 = 0.1;
Point(6) = {-0, 0, -0, 0.1};
Point(7) = {0, 0.1, -0, 0.1};
Point(8) = {0, 0.5, -0, 0.1};
Point(9) = {-0, 0.9, 0, 0.1};
Point(10) = {0, 1, 0, 0.1};
Point(11) = {0, 0, 1, 0.1};
Point(12) = {0, 0.1, 1, 0.1};
Point(16) = {0, 0.5, 1, 0.1};
Point(17) = {0, 0.9, 1, 0.1};
Point(21) = {0, 1, 1, 0.1};
Circle(1) = {7, 8, 9};
Circle(2) = {6, 8, 10};
Line(3) = {6, 7};
Line(4) = {9, 10};
Line(8) = {11, 12};
Circle(9) = {12, 16, 17};
Line(10) = {17, 21};
Circle(11) = {11, 16, 21};
Line(12) = {10, 21};
Line(13) = {9, 17};
Line(14) = {7, 12};
Line(15) = {11, 6};
Line Loop(6) = {3, 1, 4, -2};
Ruled Surface(6) = {6};
Line Loop(7) = {8, 9, 10};
Ruled Surface(7) = {7};
Line Loop(17) = {12};
Plane Surface(17) = {17};
Line Loop(19)={13,-9,-14,1};
Ruled Surface(19) = {19};
Line Loop(21) = {13, -4, -12, 10};
Plane Surface(21) = {21};
Line Loop(23) = {14, -8, 15, 3};
Plane Surface(23) = {23};
Delete {
  Surface{7};
}
Line Loop(24) = {11, -12, -2, -15};
Ruled Surface(25) = {24};
Line Loop(26) = {9, 10, -11, 8};
Ruled Surface(27) = {26};
