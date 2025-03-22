function f1 = readTxtCol(fileName,filePath,savePath,saveName,col)
    file = fullfile(filePath,fileName);
    f = fopen(file);
    if f == -1
        error(['File not found or unable to open: ', file]);
    end
    dt = textscan(f,'%s');
    % 调整格式
    % 去除标题
    dt1{1,1} = dt{1,1}(col+1:end,:);
    L = length(dt1{1,1})/col; % 行数
    for i = 1:L
        for j = 1:col
            f1(i,j)=str2double(dt1{1,1}{(i-1)*col+j,1});
        end
        if mod(i,1e4) == 0
            disp([fileName,' row:',num2str(i)]);
        end
    end
    if ~exist('f1')
        error(['the file ', fileName,' dont exist']);
    end
    eval([saveName(1:end-4),'=f1;']); % 执行字符串写的指令
    savePath = [savePath,saveName];
    save(savePath,saveName(1:end-4));
end