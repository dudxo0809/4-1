function Func = assign_2016722074(A, b)

m = size(A, 1);

i_b = size(b, 1);
j_b = size(b, 2);


% #1
% Column-wise interchange in matrix
% column 1 <-> column m
temp = A(:,m);
A(:,m) = A(:,1);
A(:,1) = temp;

disp("Num 1) ");
disp(A);


% #2
% Sort the odd row in ascending order
for i=1:m
    if rem(i,2)==1
        A(i,:) = sort(A(i,:));
    end
end

disp("Num 2) ");
disp(A);


% #3
% Add one to all diagonal term
for i=1:m
    for j=1:m
        if i==j
            A(i,j) = A(i,j) + 1;
        end
    end
end

disp("Num 3) ");
disp(A);


% #4
% Calculate inner product
% B = A * A
B = zeros([m,m]);
for i=1:m
    for j=1:m
        for k=1:m
            B(i,j) = B(i,j) + (A(i,k) * A(k,j));
        end
    end
end

disp("Num 4) ");
disp(B);

% #5
% vector u : m by 1
u = zeros([m,1]);

for i=1:m
    u(i) = b(i);
end

disp("Num 5) ");
disp(u);


% #6
% vector t = A * u
t = zeros([m,1]);

for i=1:m
    sum = 0;
    for j=1:m
        sum = sum + A(i,j) * u(i);
    end
    t(i) = sum;
end

disp("Num 6) ");
disp(t);


% #7
% Concatenate t & u (m by 2) => matrix C
C = zeros([m,2]);

for i=1:m
    C(i,1) = t(i);
    C(i,2) = u(i);
end

disp("Num 7) ");
disp(C);


% #8
% D = t * u
D = zeros([m,m]);

for i=1:m
    for j=1:m
        D(i,j) = dot(t,u);
    end
end

disp("Num 8) ");
disp(D);


% #9
% Create matrix E
% element-wise power B & D
E = zeros([m,m]);

for i=1:m
    for j=1:m
        E(i,j) = B(i,j) .^ D(i,j);
    end
end

disp("Num 9) ");
Func = E;

end