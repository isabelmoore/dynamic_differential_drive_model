track.radius = 200;
track.l_st = 900;
track.width = 15;

radius = track.radius;
l_st = track.l_st;
l_curve = pi * radius;
total_length = 2 * l_st + 2 * l_curve;
delta_s = 5;
npts = round(total_length/delta_s);
ncpts = round(l_curve / delta_s);
delta_theta = delta_s / radius;

xpath = zeros(npts,1);
ypath = zeros(npts,1);

i = 1;
while i < npts
    if xpath(i) < l_st
        if xpath(i) >= 0
            if ypath(i) < radius
                xpath(i+1) = xpath(i) + delta_s;
                ypath(i+1) = ypath(i);
            else
                xpath(i+1) = xpath(i) - delta_s;
                ypath(i+1) = ypath(i);
            end
        else
            cx = 0; cy = radius;
            rx = xpath(i) - cx; ry = ypath(i) - cy;
            tt = rotate( [rx;ry], - delta_theta); 
            xpath(i+1) = tt(1) + cx; ypath(i+1) = tt(2) + cy;
        end
    else
        cx = l_st; cy = radius;
        rx = xpath(i) - cx; ry = ypath(i) - cy;
        tt = rotate( [rx;ry], - delta_theta); 
        xpath(i+1) = tt(1) + cx; ypath(i+1) = tt(2) + cy;
    end
    i = i + 1;    
end