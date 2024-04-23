function varargout = cropmat(IX,varargin)

% clip a subset of arrays with axis-aligned minimum bounding box
%
% Syntax
%
%     [Ac,Bc,Cc,...] = cropmat(I,A,B,C,...)
%     [Ac,Bc,Cc,...] = cropmat(ix,A,B,C,...)
%
% Description 
%
%     cropmat extracts a subset of various arrays with any dimension.
%     The subset mask is the axis-aligned minimum bounding rectangle,  
%     box or hyperrectangle of true values in I or linear indices ix  
%     in A, B, C,... .
%
%     I must be a logical array. I, A, B, C, ... must all have same
%     size. cropmat supports any data type and cell arrays. 
% 
% Example (requires the image processing toolbox)
%
%     I = imread('football.jpg');
%     BW = im2bw(I);
%     Ic = cropmat(repmat(BW,[1,1,3]),I);
%     BWc= cropmat(BW,BW);
%     
%     % plot results
%     subplot(1,3,1)
%     imshow(I)
%     title('original image (I)')
%
%     subplot(1,3,2)
%     imshow(BW)
%     title('clip mask (BW)')     
%
%     subplot(1,3,3)
%     h = imshow(Ic)
%     set(h,'AlphaData',(BWc+1)/2); 
%     title('clipped image (Ic)')   
%
%
% See also: IND2SUB, SUBVOLUME
%
% Author: Wolfgang Schwanghart (w.schwanghart[at]unibas.ch)
% Date: 18/9/2009




% check number of input and output arguments
error(nargchk(2,inf,nargin))
error(nargoutchk(0, max(nargin-1,1), nargout))

% what type is the first argument
if islogical(IX);
    siz = size(IX);
    IX  = find(IX);
else
    siz = size(varargin{1});
    if max(IX)>numel(varargin{1})
        error('max(ix) is larger than the maximum number of elements in the matrices')
    end
end
 
% check if all matrices have same size
if nargin>1
    if any(cell2mat(cellfun(@(x) ~isequal(size(x),siz),...
            varargin,'uniformoutput',false)));
        error('all matrices must have same size')
    end
end

% nr of dimensions
dims = numel(siz);
k    = [1 cumprod(siz(1:end-1))];

% preallocate subsref structure
S    = substruct('()',cell(1,dims));

% subset size
sizout = zeros(1,dims);

% loop through dimensions (see ind2sub) 
% and get subscripts of minimum bounding rectangle/box/...
for r = dims:-1:1;  
    IX2       = rem(IX-1,k(r))+1;         
    subdim    = (IX-IX2)/k(r)+1; 
    S.subs{r} = min(subdim):max(subdim);
    sizout(r) = numel(S.subs{r});
    IX        = IX2;   
end

% index into all input arguments and reshape them to new size
varargout = cellfun(@(X) reshape(subsref(X,S),sizout),...
                    varargin,'uniformoutput',false);

        

