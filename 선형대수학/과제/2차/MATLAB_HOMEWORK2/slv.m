function x = slv(A, b)          % 매트랩은 먼저 선언된 함수가 파일명으로 호출시 가장 먼저 불리기 때문에 slv를 slu보다 위에 위치했습니다.
[P, L, U] = slu(A);         % slu 함수를 사용해 각각 P,L,U를 구함 [PA=LU]
[n, n] = size(A);           % 선언된 배열의 size를 얻음

b = P*b;                    % row exchange를 b에 반영 [PA=LU]를 통해 접근했기 때문
c = zeros(n, 1);            % n사이즈의 컬럼 벡터 생성

for k = 1:n                 % forward 소거(소거단계) 방정식의 미지수를 소거해 하나의 미지수에 대해 정리
    s = 0;                    % 디버그 (s를 선언하고 0으로 초기화)
    for j = 1:k-1             % c(k)이전에 L번 더 일찍 c(j)를 더함
        s = s + c(j)*L(k,j); end  % c(k)이전에 L번 더 일찍 c(j)를 더함
    c(k) = b(k) - s;
end      %c(K)를 찾고 다음 k값의 대입을 위해 s를 초기화
x = zeros(n, 1);            % 기능강화 (해 행렬 x선언. 이전엔 디폴트라 x=x'를 통해 컬럼으로 바꿔줘야 했지만 컬럼선언을 통해 바로 해 출력)
for k = n:-1:1              %x (n)에서 x(1)까지 뒤로 되짚기(대입단을 위해).
    t = 0;                    % 디버그 (t를 선언하고 0으로 초기화)
    for j = k+1:n             % 대입단계(해를 방정식에 대입해 모르는 해를 구하는 단계)
        t = t + U(k, j)*x(j);
    end %x(j)이후에 U만큼 곱해줌
    x(k) = (c(k) - t) / U(k, k);
end %pivot으로 나눠줌
if isnan(x(1)), fprintf("@@@@@@  Input 'A' is singular matrix. Please input non-singular matrix  @@@@@@") % singular matrix의 경우 연산이 제대로 진행되지 않아 'NaN'값이 들어가게됨. 이를 통해  singular matrix를 찾아낸다.
end
end                        % singualr matrix용 if문 종료
