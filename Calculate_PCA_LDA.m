% Brain State Trajectories
%
% Using PCA and LDA
%
% 
%
% Created by Eli Muller 2020

% Load data
X = []; % observation x variable


% ------------- PCA
[pc_vec, pc_val] = pca(X); % PCA of concatenated data
n_pcs = 10; % Choose data dimensionality


%------------- LDA

% Define data classes (divisions)
X1 = pc_val(1:div1,1:n_pcs);
X2 = pc_val(div1+1:div2,1:n_pcs);
X3 = pc_val(div2+1:div3,1:n_pcs);
X4 = pc_val(div4+1:div4,1:n_pcs);

% Class sizes
N1 = size(X1,1);
N2 = size(X2,1);
N3 = size(X3,1);
N4 = size(X4,1);

% Class mean
Mu1 = mean(X1,1)';
Mu2 = mean(X2,1)';
Mu3 = mean(X3,1)';
Mu4 = mean(X4,1)';

% Data mean
Mu = (Mu1 + Mu2 + Mu3 + Mu4)./4;

% Between-class scatter martrix
Sb = N1.*(Mu1 - Mu)*(Mu1 - Mu)' + N2.*(Mu2 - Mu)*(Mu2 - Mu)' + N3.*(Mu3 - Mu)*(Mu3 - Mu)'+ N4.*(Mu4 - Mu)*(Mu4 - Mu)';

% Within-class scatter matrices
S1 = cov(X1);
S2 = cov(X2);
S3 = cov(X3);
S4 = cov(X4);


Sw = S1 + S2 + S3 + S4; % Aggregate within-class scatter
inv_Sw = inv(Sw);

G = inv_Sw*Sw; % Check inverse

[eig_vec, eig_val] = eig(inv_Sw*Sb); % Eigendecomposition
D = real(diag(eig_val));
[~, eig_order] = sort(D,'descend');
lda_eig = eig_vec(:,eig_order); % Sort eigenvalues in descending order



% ---------------- Orthonormalize
% Not neccessary but but can be useful for eigenvector interpretation and
% data reconstruction

X = lda_eig;
% Modified Gram-Schmidt. [Q,R] = mgs(X);
% G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
[n,p] = size(X);
Q = zeros(n,p);
R = zeros(p,p);
Q(:,1) = X(:,1);
for kk = 2:p
    Q(:,kk) = X(:,kk);
    for ii = 1:kk-1
      R(ii,kk) = Q(:,ii)'*Q(:,kk);
      Q(:,kk) = Q(:,kk) - R(ii,kk)*Q(:,ii);
    end
    R(kk,kk) = norm(Q(:,kk))';
    Q(:,kk) = Q(:,kk)/R(kk,kk);
end
lda_eig = Q;


% LDA vectors in original space
lda_orig_space = pc_vec(:,1:n_pcs)*lda_eig;

% Project data into LDA subspace
x_lda_proj = pc_val*lda_eig;







