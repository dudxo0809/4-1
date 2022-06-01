function [P, L, U] = slu(A)     % 매트랩은 먼저 선언된 함수가 파일명으로 호출시 가장 먼저 불리기 때문에 slu를 slv보다 아래에 위치했습니다.
[n, n] = size(A);               % n에 A행렬의 가로 세로 개수 대입 (항상 정사각 행렬임으로 n,n으로 대입 하여도 무관)
P = eye(n, n);                  % (n,n)크기의 단위행렬 P 생성
L = eye(n, n);                  % (n,n)크기의 단위행렬 L 생성
U = zeros(n, n);                % (n,n)크기의 영행렬 U 생성

tol = 1.e-6; pivot_k=1;                 % tol = 0.000001 차후에 값이 음수인지 확인하기 위한 비교 변수로 사용됨
    for k = 1:n                         % LU factorization 과정 시작
        chk_len = k-1;                  % 길이 조정을 위한 체크 변수 선언
       [r, pivot_k] = check_pivot(A(k:n, pivot_k:n), k, pivot_k, tol);    % row exchange를 위해 check_pivot 함수 구현 n값이 증가하면서 넘어가는 A행렬 자체도 감소 n회 만큼 반복됨을 확실히!
       if ne(r,k)                                                   % r과 k가 같지 않은 경우 row exchage가  구현되야함
          A([r k], 1:n) = A([k r], 1:n);                            % A 행렬에 k-r 값 변환 (해당값을 기준으로 row exchange 진행)
          if k > 1, L([r k], 1:chk_len) = L([k r], 1:chk_len); 
          end          % k가 1이상일때 (값이 변화함으로)L 행렬에 k-r 값 변환 
          P([r k], 1:n) = P([k r], 1:n); 
       end                        % P를통한 row exchage 진행
       if abs(A(k, pivot_k)) >= tol   % A(k,k)->A(k,pivot_k)로 변환 k값 변환으로 열 변경 문제 접근
          for i = k+1:n               % 각 k값 저장을 위한 반복문
             L(i, k) = A(i, pivot_k) / A(k, pivot_k); % 곱해지는 값인 k가 L행렬에 들어가게됨 (LU팩토라이제이션 기본 정의)
             for j = k+1:n            %k행과 k열에 대한  소거법 적용
                A(i,j) = A(i, j) - A(k, j)*L(i, k); 
             end 
          end 
       end % 요소가 바뀌었지만 그대로 A 
       for j = k:n                    % 대입하여 U 행렬 생성을 위한 반복문
          U(k, j) = A(k, j) * (abs(A(k, j)) >= tol); 
       end % A요소가 0이 아닐경우 U에 각각의 값을 대입하여 U행렬 생성
       
       if pivot_k < n
           pivot_k = pivot_k+1; 
       end
    end
end      % 밖에 있을때(check_pivot에서 -1한걸 취소)
