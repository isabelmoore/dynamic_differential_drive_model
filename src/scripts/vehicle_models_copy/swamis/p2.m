
xpath = X1p;
ypath = Y1p;
theta = psi1p;
figure
plot(xpath,ypath,'.r');
hold on
h = animatedline;
%axis([min(xpath) max(xpath) min(ypath) max(ypath)])
axis([min([xpath;ypath]) max([xpath;ypath]) min([xpath;ypath]) max([xpath;ypath])])
for i = 1:length(xpath)
    x = xpath(i);
    y = ypath(i);
    addpoints(h,x,y)
    width = 1;
    Lcar = 2;
    car = [-Lcar/2 -width/2; -Lcar/2 width/2; Lcar/2 width/2; Lcar/2 -width/2];
    rcar = rotate(car', theta(i))';
    a = polyshape(rcar+ [x,y]);
    ap = plot(a);
    drawnow limitrate
    ap.FaceColor = 'none';
    drawnow limitrate
    pause(0.05);
end

hold off