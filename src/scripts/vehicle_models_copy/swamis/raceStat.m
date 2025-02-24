function raceStats = raceStat(X,Y,t,track)
%========================================================
% 
% Usage: rs = raceStat(X,Y,t,track)
%
% Inputs:   X, Y and t are X, Y and t coordinates from your vehicle
%           simulations. 
%           track is a structure defining the track. It has the
%           following fields: track, width and l_st (length of the
%           straight section
%           track is a structure of with fields "width" (width of the 
%                  track), "l_st" (length of the straight away), and 
%                  "radius" (radius of the curved section)
%
% Outputs: raceStats is a structure with the following fields:
%
%   loops - this is an integer that tells how many loops around
%              the track the vehicle has gone around
%   tloops - this is an array that tells you the time(s) that the
%              start line was crossed 
%   lefftTrack - this has the X,Y and t values when the vehicle
%              went outside the track
%   
%========================================================

prev_section = 6;
loops = -1;
j = 0;
k = 0;
Xerr = [];
Yerr = [];
terr = [];
for i = 1:length(X)
    if X(i) < track.l_st
        if X(i) >= 0
            if Y(i) < track.radius
                section = 1;
            else
                section = 4;
            end
        else
            if Y(i) < track.radius
                section = 6;
            else
                section = 5;
            end
        end
    else
        if Y(i) < track.radius
            section = 2;
        else
            section = 3;
        end
    end
    if ((prev_section == 6) && (section == 1))
        loops = loops  + 1;
        j = j+1;
        tloops(j) = t(i);
    end
    prev_section = section;
    if ~insideTrack(X(i),Y(i),section,track)
        k = k+1;
        Xerr(k) = X(i);
        Yerr(k) = Y(i);
        terr(k) = t(i);
    end
end
raceStats.loops = loops;
raceStats.tloops = tloops;
raceStats.leftTrack.X = Xerr;
raceStats.leftTrack.Y = Yerr;
raceStats.leftTrack.t = terr;
end

function yesorno = insideTrack(x,y,section,track)
switch section
    case 1
        if ((y < (0.0 + track.width)) && (y > (0.0 - track.width))) 
            yesorno = 1;
        else
            yesorno = 0;
        end
    case {2, 3}
        rad = sqrt((x - track.l_st)^2 + (y - track.radius)^2);
        if ((rad < track.radius + track.width) && ...
                (rad > track.radius - track.width))
            yesorno = 1;
        else
            yesorno = 0;
        end
    case 4
        if ((y < (2 * track.radius + track.width)) && ...
                (y > (2 * track.radius - track.width))) 
            yesorno = 1;
        else
            yesorno = 0;
        end        
    case {5, 6}
        rad = sqrt((x - 0.0)^2 + (y - track.radius)^2);
        if ((rad < track.radius + track.width) && ...
                (rad > track.radius - track.width))
            yesorno = 1;
        else
            yesorno = 0;
        end
    otherwise
        print("error");
end
end