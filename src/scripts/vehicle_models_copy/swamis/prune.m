function [t2,X2] = prune(t1,X1,minDT)
it2 = find(diff(t1)>minDT);
t2 = t1(it2);
X2 = X1(it2,:);
end