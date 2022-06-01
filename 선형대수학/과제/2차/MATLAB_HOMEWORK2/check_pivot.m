function [k, pivot_k] = check_pivot(A, k, pivot_k, tol)     % row exchage를 위해 slu에서 호출 되는 check_pivot 함수
    [n,  n] = size(A);              % n에 배열의 행,렬 배정 (정사각행렬이기 때문에 같다)
    r = find(abs(A(:)) > tol);      % r에 ifnd를 통해 A배열안에 0보다 큰값이 있는곳을 체크하여 저장 (0을 제외하는 과정)
    if isempty(r)
        return
    end      % isempty함수를 통해 r 배열이 비어있는지 확인 비어있다면 종료
    r = r(1);                       % 정상 초기화
    j = fix((r-1)/n)+1;             % r-1에서 n으로 나눈후 소수점 단위 삭제
    k = k+r-(j-1)*n-1;              % k값 (가곱셈값;multipliers) 계산후 할당
    pivot_k = pivot_k+j-1; 
end      % pivot_k에 j값을 저장한 이후 -1 (밖일시 slu에서 다시 조정)