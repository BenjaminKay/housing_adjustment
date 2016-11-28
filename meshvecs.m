function Vecs = meshvecs(x,y,z) 
%Like mesh grid, but with no error checking and automatically stacks
%everything into vectors in one long matrix
x=x';
z=reshape(z(:),[1 1 numel(z)]);
Vecs = [reshape(x(ones(numel(y),1),:,ones(numel(z),1)),[],1),reshape(y(:,ones(1,numel(x)),ones(numel(z),1)),[],1),reshape(z(ones(numel(y),1),ones(numel(x),1),:) ,[],1)];