clc

test_x=load('test_x.txt');
test_y=load('test_y.txt');
test_s=load('test_s.txt');

size(test_x)

[c,m,C]=getABfit(test_x,test_y,test_s)

