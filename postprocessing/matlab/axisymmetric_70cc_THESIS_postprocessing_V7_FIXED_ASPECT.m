%% ================================================================
%  AXISYMMETRIC FEniCS DG0 CONCENTRATION POST-PROCESSING
%
%  For the final axisymmetric FEniCS model:
%
%       transformed coordinate:  s = r^2/2
%       physical radius:          r = sqrt(2s)
%
%  The concentration field c is DG0:
%       ONE concentration value per triangular CELL,
%       NOT one value per mesh vertex.
%
%  Typical final mesh:
%       vertices = 6366
%       cells    = 12546
%
%  Therefore a dataset containing 12546 values is CORRECT.
%
%  Main outputs:
%
%   1. Clean mirrored physical r-z concentration snapshots
%          c = 0 fresh water -> white
%          c = 1 salt water  -> black
%
%   2. Clean axisymmetric mean concentration profiles
%
%          cbar(z,t)
%            = (1/S) integral_0^S c(s,z,t) ds
%
%      where S = R^2/2.
%
%      Because ds = r dr, this is exactly the physical
%      cross-sectional radial average
%
%          cbar
%            = (2/R^2) integral_0^R c(r,z,t) r dr.
%
%      The DG0 field is averaged directly using triangle areas in
%      horizontal s-z slabs. No nodal interpolation is used for the
%      quantitative profiles.
%
%   3. Time-height diagram of mean concentration
%
%   4. .mat and .csv files containing the averaged profiles
%
%  High-resolution interpolation is used ONLY for rendering the
%  concentration PNGs.


clear
close all
clc


%% ================================================================
%  USER SETTINGS


xdmfFile = 'c.xdmf';
h5File   = 'c.h5';

% Physical velocity written by the FEniCS model.
velocityXdmfFile = 'u_physical.xdmf';
velocityH5File   = 'u_physical.h5';

% Thesis streamline times.
streamlineTimes = [200 500];

% Streamline plotting grid.
nStreamR = 301;
nStreamZ = 181;
streamlineDensity = 1.25;

R = 0.30;                  % physical tank radius [m]
H = 0.30;                  % tank height [m]

resultsFolder = 'FEniCS_axisymmetric_THESIS_results';

% Save every available field for checking the full evolution.
saveAllFields = true;

% Main six-panel thesis high-flow sequence.
% Requested times beyond the currently available output are ignored.
thesisFieldTimes = [50 150 250 350 450 550];

% Times used on the thesis mean-profile figures.
profileTimes = [50 100 200 300 400 500];

% Number of vertical bins used for the QUANTITATIVE area-weighted
% DG0 mean profiles.
nProfileBins = 120;

% Display grid only. Does not affect quantitative averages.
nPlotR = 1200;
nPlotZ = 600;

% CONTRAST FIELD:
%
% Plot the departure from ambient concentration,
%
%       delta_c = 1-c
%
% using only the range 0 <= delta_c <= contrastFreshMax.
% Values above contrastFreshMax are saturated white.
%
% This is the useful contrast view used to reveal the weak upper
% stratification that is almost invisible on the full c = 0--1 scale.
% MAIN THESIS CONTRAST:
%
%       0.98 <= c <= 1
%
% This corresponds to 0 <= 1-c <= 0.02.  It is less saturated than
% the old 0.99--1 view and is the preferred fixed scale for comparing
% the 20 and 70 cc/min calculations.
contrastFreshMax = 0.02;

% Keep the stronger 0.99--1 contrast as a diagnostic only.
diagnosticContrastFreshMax = 0.01;

% Additional broader thesis-ready contrast:
%
%       0.97 <= c <= 1
%
% This shows the broader concentration structure with less
% saturation than the 0.98--1 view, while still retaining good
% contrast in the weakly freshened upper region.
broadThesisContrastFreshMax = 0.03;

% Save a separate clean profile for each selected time?
saveIndividualProfiles = true;

profileResolution = 600;

% Clean profile figures are deliberately portrait, matching the other
% numerical profile figures used in the thesis.
profileFigureSize = [1000 1600];

% Exact thesis profile output geometry.
% 5 x 8 inches at 200 dpi gives 1000 x 1600 px.
profilePaperSizeInches = [5 8];
profileDPI = 200;


%% ================================================================
%  CHECK INPUT FILES


if ~isfile(xdmfFile)
    error('Could not find %s in the current folder.',xdmfFile)
end

if ~isfile(h5File)
    error('Could not find %s in the current folder.',h5File)
end


%% ================================================================
%  OUTPUT FOLDERS


if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder)
end

fieldFolder = fullfile(resultsFolder,'01_raw_fields_all');
contrastFolder = fullfile(resultsFolder,'02_contrast_098_100_all');
thesisFieldFolder = fullfile(resultsFolder,'03_thesis_selected_contrast_098_100');
diagnosticFolder = fullfile(resultsFolder,'04_selected_diagnostic_contrast_099_100');
broadContrastFolder = fullfile(resultsFolder,'05_thesis_selected_contrast_097_100');
profileFolder = fullfile(resultsFolder,'06_profiles');
thesisReadyFolder = fullfile(resultsFolder,'07_THESIS_READY');
streamlineFolder = fullfile(resultsFolder,'08_velocity_streamlines');
analysisFolder = fullfile(resultsFolder,'09_analysis');

if ~exist(fieldFolder,'dir')
    mkdir(fieldFolder)
end

if ~exist(contrastFolder,'dir')
    mkdir(contrastFolder)
end

if ~exist(thesisFieldFolder,'dir')
    mkdir(thesisFieldFolder)
end

if ~exist(diagnosticFolder,'dir')
    mkdir(diagnosticFolder)
end

if ~exist(broadContrastFolder,'dir')
    mkdir(broadContrastFolder)
end

if ~exist(profileFolder,'dir')
    mkdir(profileFolder)
end

if ~exist(thesisReadyFolder,'dir')
    mkdir(thesisReadyFolder)
end

if ~exist(streamlineFolder,'dir')
    mkdir(streamlineFolder)
end

if ~exist(analysisFolder,'dir')
    mkdir(analysisFolder)
end


%% ================================================================
%  READ SAVED TIMES FROM XDMF


fprintf('Reading %s...\n',xdmfFile)

xmlText = fileread(xdmfFile);

tokens = regexp( ...
    xmlText, ...
    '<Time\s+Value="([^"]+)"', ...
    'tokens');

nTimes = numel(tokens);

if nTimes == 0
    error('No saved time values were found in %s.',xdmfFile)
end

times = zeros(nTimes,1);

for k = 1:nTimes
    times(k) = str2double(tokens{k}{1});
end

fprintf('Found %d saved timesteps.\n',nTimes)
fprintf('First time = %.6g s\n',times(1))
fprintf('Last time  = %.6g s\n\n',times(end))


%% ================================================================
%  READ AXISYMMETRIC TRANSFORMED MESH


geometry = h5read( ...
    h5File, ...
    '/Mesh/0/mesh/geometry');

topology = h5read( ...
    h5File, ...
    '/Mesh/0/mesh/topology');


% Geometry -> Nvertices x 2
if size(geometry,2) == 2

    % already correct

elseif size(geometry,1) == 2

    geometry = geometry';

else

    error( ...
        'Unexpected geometry dimensions: %d x %d.', ...
        size(geometry,1), ...
        size(geometry,2))

end


% Triangle topology -> Ncells x 3
if size(topology,2) == 3

    % already correct

elseif size(topology,1) == 3

    topology = topology';

else

    error( ...
        'Expected triangular topology, but topology is %d x %d.', ...
        size(topology,1), ...
        size(topology,2))

end


topology = double(topology);

% FEniCS connectivity is zero-based. MATLAB is one-based.
if min(topology(:)) == 0
    topology = topology + 1;
end


sNode = geometry(:,1);
zNode = geometry(:,2);

nNodes = size(geometry,1);
nCells = size(topology,1);

fprintf('Axisymmetric transformed mesh:\n')
fprintf('    vertices = %d\n',nNodes)
fprintf('    cells    = %d\n',nCells)
fprintf('    s range  = %.6g to %.6g\n',min(sNode),max(sNode))
fprintf('    z range  = %.6g to %.6g m\n\n',min(zNode),max(zNode))


%% ================================================================
%  PRECOMPUTE DG0 CELL GEOMETRY


v1 = topology(:,1);
v2 = topology(:,2);
v3 = topology(:,3);


% Cell centroids in transformed s-z coordinates
sCell = ...
    (sNode(v1) + sNode(v2) + sNode(v3)) / 3;

zCell = ...
    (zNode(v1) + zNode(v2) + zNode(v3)) / 3;


% Triangle area in s-z coordinates
cellArea = 0.5 * abs( ...
    (sNode(v2)-sNode(v1)).*(zNode(v3)-zNode(v1)) ...
    - ...
    (sNode(v3)-sNode(v1)).*(zNode(v2)-zNode(v1)) );


if any(cellArea <= 0)
    error('Mesh contains zero-area or invalid triangles.')
end


S = R^2/2;

fprintf('Expected transformed width S = R^2/2 = %.8g\n',S)
fprintf('Mesh transformed width        = %.8g\n\n',max(sNode)-min(sNode))


%% ================================================================
%  VERTICAL SLABS FOR QUANTITATIVE MEAN PROFILES
%
%  Each DG0 triangle is assigned by its centroid to a horizontal
%  slab. Within each slab:
%
%      cbar = sum(c_K * |K|) / sum(|K|)
%
%  Since |K| is area in ds dz, this approximates
%
%      (1/S) integral c ds
%
%  without pretending the DG0 values are nodal data.


zEdges = linspace(0,H,nProfileBins+1);
zAverage = 0.5*(zEdges(1:end-1)+zEdges(2:end));
zAverage = zAverage(:);

cellBin = discretize(zCell,zEdges);

binArea = accumarray( ...
    cellBin(~isnan(cellBin)), ...
    cellArea(~isnan(cellBin)), ...
    [nProfileBins 1], ...
    @sum, ...
    0);

if any(binArea == 0)
    warning( ...
        ['Some vertical profile bins contain no cell centroids. ' ...
         'Their values will be filled by interpolation.'])
end


%% ================================================================
%  HIGH-RESOLUTION PHYSICAL r-z DISPLAY GRID
%
%  The actual numerical coordinate is s. For plotting we evaluate
%  at s = r^2/2 and mirror about r=0.


rPlot = linspace(-R,R,nPlotR);
zPlot = linspace(0,H,nPlotZ);

[Rplot,Zplot] = meshgrid(rPlot,zPlot);

Splot = 0.5 * abs(Rplot).^2;


%% ================================================================
%  SAVED FIELD / PROFILE INDICES


if saveAllFields

    % Save every field that actually exists in the XDMF/HDF5 output.
    fieldIdx = 1:nTimes;

else

    fieldTimesUse = fieldTimes( ...
        fieldTimes >= times(1) ...
        & ...
        fieldTimes <= times(end));

    fieldIdx = zeros(size(fieldTimesUse));

    for j = 1:numel(fieldTimesUse)
        [~,fieldIdx(j)] = min(abs(times-fieldTimesUse(j)));
    end

    fieldIdx = unique(fieldIdx,'stable');

end


profileTimesUse = profileTimes( ...
    profileTimes >= times(1) ...
    & ...
    profileTimes <= times(end));


if isempty(profileTimesUse)
    profileTimesUse = times(end);
end


profileIdx = zeros(size(profileTimesUse));

for j = 1:numel(profileTimesUse)
    [~,profileIdx(j)] = min(abs(times-profileTimesUse(j)));
end

profileIdx = unique(profileIdx,'stable');


fprintf('Concentration snapshots: saving ALL %d saved fields.\n', ...
    numel(fieldIdx))

fprintf('First saved field = %.6g s\n',times(fieldIdx(1)))
fprintf('Last saved field  = %.6g s\n',times(fieldIdx(end)))

fprintf('\nMean profiles will use:\n')
fprintf('    %.6g s\n',times(profileIdx))
fprintf('\n')

% Selected thesis field times.
thesisFieldTimesUse = thesisFieldTimes( ...
    thesisFieldTimes >= times(1) ...
    & ...
    thesisFieldTimes <= times(end));

if isempty(thesisFieldTimesUse)
    thesisFieldTimesUse = times(end);
end

thesisFieldIdx = zeros(size(thesisFieldTimesUse));

for j = 1:numel(thesisFieldTimesUse)
    [~,thesisFieldIdx(j)] = min(abs(times-thesisFieldTimesUse(j)));
end

thesisFieldIdx = unique(thesisFieldIdx,'stable');

fprintf('Selected thesis field times:\n')
fprintf('    %.3f s\n',times(thesisFieldIdx))
fprintf('\n')


%% ================================================================
%  PROFILE COLOURS


if times(end) > times(1)

    timeFraction = ...
        (times-times(1)) ...
        / ...
        (times(end)-times(1));

else

    timeFraction = zeros(size(times));

end


profileColors = [ ...
    1 - 0.35*timeFraction, ...
    0.85*(1-timeFraction), ...
    0.85*(1-timeFraction) ...
];


%% ================================================================
%  STORAGE


cBarAll = nan(nProfileBins,nTimes);

% Volume-mean concentration.  The constant 2*pi factor introduced by
% revolving the transformed domain cancels in the normalised mean.
cVolumeMean = nan(nTimes,1);
freshVolumeFraction = nan(nTimes,1);


%% ================================================================
%  PROCESS EVERY SAVED DG0 FIELD


for k = 1:nTimes

    t = times(k);

    dataset = sprintf('/VisualisationVector/%d',k-1);

    fprintf( ...
        'Processing %3d / %3d   t = %.6g s\n', ...
        k,nTimes,t)


    %% ------------------------------------------------------------
    %  READ DG0 CONCENTRATION


    c = h5read(h5File,dataset);
    c = double(c(:));


    if numel(c) ~= nCells

        error( ...
            ['Dataset %s has %d values. The axisymmetric DG0 mesh ' ...
             'has %d cells and %d vertices. This postprocessor ' ...
             'expects one concentration value per CELL.'], ...
            dataset, ...
            numel(c), ...
            nCells, ...
            nNodes)

    end


    if any(~isfinite(c))

        error( ...
            'Dataset %s contains NaN or Inf values.', ...
            dataset)

    end


    %% ------------------------------------------------------------
    %  VOLUME-MEAN CONCENTRATION


    cVolumeMean(k) = ...
        sum(c.*cellArea) ...
        / ...
        sum(cellArea);

    freshVolumeFraction(k) = 1-cVolumeMean(k);


    %% ------------------------------------------------------------
    %  QUANTITATIVE AXISYMMETRIC MEAN
    %
    %  Area weighting is performed directly on DG0 cell values.


    weightedSum = accumarray( ...
        cellBin(~isnan(cellBin)), ...
        c(~isnan(cellBin)).*cellArea(~isnan(cellBin)), ...
        [nProfileBins 1], ...
        @sum, ...
        0);


    cBar = nan(nProfileBins,1);

    validBin = binArea > 0;

    cBar(validBin) = ...
        weightedSum(validBin) ...
        ./ ...
        binArea(validBin);


    % Fill only rare empty centroid bins for plotting continuity.
    if any(~validBin)

        cBar = fillmissing( ...
            cBar, ...
            'linear', ...
            'EndValues','nearest');

    end


    cBarAll(:,k) = cBar;


    %% ------------------------------------------------------------
    %  CLEAN PHYSICAL r-z CONCENTRATION SNAPSHOT
    %
    %  Interpolation is DISPLAY ONLY.


    if ismember(k,fieldIdx)

        Fdisplay = scatteredInterpolant( ...
            sCell, ...
            zCell, ...
            c, ...
            'linear', ...
            'nearest');


        Cplot = Fdisplay( ...
            Splot, ...
            Zplot);


        Cplot = max(0,min(1,Cplot));



        % RAW CONCENTRATION FIELD
        %
        % c = 0 fresh -> white
        % c = 1 salt  -> black


        % imwrite places matrix row 1 at the TOP of the image.
        % zPlot runs bottom -> top, so flip vertically before saving.
        imageGray = uint8( ...
            round(255*flipud(1-Cplot)));


        timeTag = sprintf('%07.3fs',t);
        timeTag = strrep(timeTag,'.','p');


        imwrite( ...
            imageGray, ...
            fullfile( ...
                fieldFolder, ...
                ['concentration_' timeTag '.png']));



        % CONTRAST-ENHANCED FIELD
        %
        % delta_c = 1-c
        %
        % delta_c = 0                 -> black (ambient)
        % delta_c >= contrastFreshMax -> white
        %
        % This deliberately saturates the fresh plume so that
        % the much weaker fresh-water content in the upper
        % stratified region becomes visible.


        deltaC = 1-Cplot;

        contrastPlot = ...
            deltaC ...
            / ...
            contrastFreshMax;

        contrastPlot = max( ...
            0, ...
            min(1,contrastPlot));


        % Same physical orientation correction as the raw field.
        contrastGray = uint8( ...
            round(255*flipud(contrastPlot)));


        imwrite( ...
            contrastGray, ...
            fullfile( ...
                contrastFolder, ...
                ['contrast_098_100_' timeTag '.png']));



        % Selected thesis copy on the SAME fixed 0.98--1 scale


        if ismember(k,thesisFieldIdx)

            imwrite( ...
                contrastGray, ...
                fullfile( ...
                    thesisFieldFolder, ...
                    ['contrast_098_100_' timeTag '.png']));


            % Stronger 0.99--1 view retained only as a diagnostic.
            diagnosticPlot = ...
                deltaC ...
                / ...
                diagnosticContrastFreshMax;

            diagnosticPlot = max(0,min(1,diagnosticPlot));

            diagnosticGray = uint8( ...
                round(255*flipud(diagnosticPlot)));

            imwrite( ...
                diagnosticGray, ...
                fullfile( ...
                    diagnosticFolder, ...
                    ['contrast_099_100_' timeTag '.png']));


            % BROADER THESIS CONTRAST: 0.97 <= c <= 1
            broadContrastPlot = ...
                deltaC ...
                / ...
                broadThesisContrastFreshMax;

            broadContrastPlot = max(0,min(1,broadContrastPlot));

            broadContrastGray = uint8( ...
                round(255*flipud(broadContrastPlot)));

            imwrite( ...
                broadContrastGray, ...
                fullfile( ...
                    broadContrastFolder, ...
                    ['contrast_097_100_' timeTag '.png']));


            % Convenient thesis-ready copies of both fixed scales.
            imwrite( ...
                contrastGray, ...
                fullfile( ...
                    thesisReadyFolder, ...
                    ['FIELD_098_100_' timeTag '.png']));

            imwrite( ...
                broadContrastGray, ...
                fullfile( ...
                    thesisReadyFolder, ...
                    ['FIELD_097_100_' timeTag '.png']));

        end

    end

end


%% ================================================================
%  SAVE INDIVIDUAL CLEAN MEAN PROFILES


if saveIndividualProfiles

    for j = 1:numel(profileIdx)

        k = profileIdx(j);
        t = times(k);


        fig = figure( ...
            'Color','w', ...
            'Visible','off', ...
            'Units','pixels', ...
            'Position',[100 100 profileFigureSize]);


        ax = axes( ...
            'Parent',fig, ...
            'Units','normalized', ...
            'Position',[0 0 1 1]);


        plot( ...
            ax, ...
            cBarAll(:,k), ...
            zAverage/H, ...
            'LineWidth',4, ...
            'Color',profileColors(k,:));


        xlim(ax,[0.98 1])
        ylim(ax,[0 1])

        axis(ax,'off')


        timeTag = sprintf('%07.3fs',t);
        timeTag = strrep(timeTag,'.','p');


        % Save at an exact 5:8 portrait aspect ratio.
        saveExactPortraitFigure( ...
            fig, ...
            fullfile( ...
                profileFolder, ...
                ['axisymmetric_average_' timeTag '.png']), ...
            fullfile( ...
                profileFolder, ...
                ['axisymmetric_average_' timeTag '.pdf']), ...
            profilePaperSizeInches, ...
            profileDPI);


        close(fig)

    end

end


%% ================================================================
%  COMBINED CLEAN MEAN-PROFILE FIGURE


fig = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Units','pixels', ...
    'Position',[100 100 profileFigureSize]);


ax = axes( ...
    'Parent',fig, ...
    'Units','normalized', ...
    'Position',[0 0 1 1]);


hold(ax,'on')


for j = 1:numel(profileIdx)

    k = profileIdx(j);

    plot( ...
        ax, ...
        cBarAll(:,k), ...
        zAverage/H, ...
        'LineWidth',4, ...
        'Color',profileColors(k,:));

end


xlim(ax,[0.98 1])
ylim(ax,[0 1])

axis(ax,'off')


% Save exact portrait copies.
saveExactPortraitFigure( ...
    fig, ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_selected_times_clean.png'), ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_selected_times_clean.pdf'), ...
    profilePaperSizeInches, ...
    profileDPI);


saveExactPortraitFigure( ...
    fig, ...
    fullfile( ...
        thesisReadyFolder, ...
        'ACTUAL_PROFILES_selected_clean.png'), ...
    fullfile( ...
        thesisReadyFolder, ...
        'ACTUAL_PROFILES_selected_clean.pdf'), ...
    profilePaperSizeInches, ...
    profileDPI);


close(fig)


%% ================================================================
%  TIME-HEIGHT DIAGRAM


fig = figure( ...
    'Color','w', ...
    'Visible','off');


ax = axes(fig);


imagesc( ...
    ax, ...
    times, ...
    zAverage/H, ...
    cBarAll);


set(ax,'YDir','normal')


xlabel( ...
    ax, ...
    'Time, $t$ (s)', ...
    'Interpreter','latex')


ylabel( ...
    ax, ...
    'Normalised height, $z/H$', ...
    'Interpreter','latex')


clim(ax,[0.98 1])

% c = 0 fresh = white
% c = 1 salt  = black
colormap(ax,flipud(gray(256)))


cb = colorbar(ax);


ylabel( ...
    cb, ...
    'Axisymmetric mean concentration, $\overline{c}$', ...
    'Interpreter','latex')


set(ax, ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'Box','on')


exportgraphics( ...
    fig, ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_time_height.png'), ...
    'Resolution',300)


exportgraphics( ...
    fig, ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_time_height.pdf'), ...
    'ContentType','vector')


close(fig)


%% ================================================================
%  INDEPENDENTLY NORMALISED PROFILE SHAPES
%
%  Shape comparison only:
%
%      c* = (cbar-min(cbar))/(max(cbar)-min(cbar))
%
%  A clean no-axis version is saved in THESIS_READY.


cBarNormalisedSelected = nan(nProfileBins,numel(profileIdx));



% CLEAN THESIS-READY VERSION


fig = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Units','pixels', ...
    'Position',[100 100 profileFigureSize]);

ax = axes( ...
    'Parent',fig, ...
    'Units','normalized', ...
    'Position',[0 0 1 1]);

hold(ax,'on')


for j = 1:numel(profileIdx)

    k = profileIdx(j);

    thisProfile = cBarAll(:,k);

    cMin = min(thisProfile);
    cMax = max(thisProfile);

    if cMax-cMin < 1e-14
        continue
    end

    cNorm = ...
        (thisProfile-cMin) ...
        / ...
        (cMax-cMin);

    cBarNormalisedSelected(:,j) = cNorm;

    plot( ...
        ax, ...
        cNorm, ...
        zAverage/H, ...
        'LineWidth',4, ...
        'Color',profileColors(k,:));

end


xlim(ax,[0 1])
ylim(ax,[0 1])
axis(ax,'off')


% Save the clean normalised profile at the same exact 5:8 ratio.
normalisedProfilePNG = fullfile( ...
    thesisReadyFolder, ...
    'NORMALISED_PROFILES_selected_clean.png');

normalisedProfilePDF = fullfile( ...
    thesisReadyFolder, ...
    'NORMALISED_PROFILES_selected_clean.pdf');

saveExactPortraitFigure( ...
    fig, ...
    normalisedProfilePNG, ...
    normalisedProfilePDF, ...
    profilePaperSizeInches, ...
    profileDPI);

savedInfo = imfinfo(normalisedProfilePNG);

fprintf( ...
    'Saved normalised profile PNG = %d x %d pixels.\n', ...
    savedInfo.Width, ...
    savedInfo.Height);


close(fig)



% LABELLED ANALYSIS VERSION


fig = figure( ...
    'Color','w', ...
    'Visible','off');

ax = axes(fig);
hold(ax,'on')


for j = 1:numel(profileIdx)

    k = profileIdx(j);
    cNorm = cBarNormalisedSelected(:,j);

    if all(isnan(cNorm))
        continue
    end

    plot( ...
        ax, ...
        cNorm, ...
        zAverage/H, ...
        'LineWidth',2, ...
        'Color',profileColors(k,:), ...
        'DisplayName',sprintf('$t=%.0f~\\mathrm{s}$',times(k)));

end


xlabel( ...
    ax, ...
    'Independently normalised concentration', ...
    'Interpreter','latex')

ylabel( ...
    ax, ...
    '$z/H$', ...
    'Interpreter','latex')

xlim(ax,[0 1])
ylim(ax,[0 1])

legend( ...
    ax, ...
    'Location','best', ...
    'Interpreter','latex')

set(ax, ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'Box','on')


exportgraphics( ...
    fig, ...
    fullfile( ...
        profileFolder, ...
        'mean_profiles_independently_normalised_labelled.pdf'), ...
    'ContentType','vector')


exportgraphics( ...
    fig, ...
    fullfile( ...
        profileFolder, ...
        'mean_profiles_independently_normalised_labelled.png'), ...
    'Resolution',300)


close(fig)


%% ================================================================
%  VOLUME-MEAN FRESHENING
%
%  Useful for checking how much the full-domain concentration is
%  still changing after approximately 500 s.


fig = figure( ...
    'Color','w', ...
    'Visible','off');

ax = axes(fig);

plot( ...
    ax, ...
    times, ...
    freshVolumeFraction, ...
    'k-', ...
    'LineWidth',2)

xlabel( ...
    ax, ...
    'Time, $t$ (s)', ...
    'Interpreter','latex')

ylabel( ...
    ax, ...
    '$1-\langle c\rangle_V$', ...
    'Interpreter','latex')

xlim(ax,[times(1) times(end)])

set(ax, ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'Box','on')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'volume_mean_freshening.pdf'), ...
    'ContentType','vector')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'volume_mean_freshening.png'), ...
    'Resolution',300)

close(fig)


%% ================================================================
%  THESIS CONTRAST SCALE REFERENCE: 0.98--1


fig = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Units','pixels', ...
    'Position',[100 100 800 180]);

ax = axes( ...
    'Parent',fig, ...
    'Position',[0.08 0.55 0.84 0.18]);

imagesc( ...
    ax, ...
    linspace(0.98,1,512))

set(ax,'YTick',[])
clim(ax,[0.98 1])
colormap(ax,flipud(gray(256)))

xlabel( ...
    ax, ...
    'Concentration, $c$', ...
    'Interpreter','latex')

set(ax, ...
    'FontSize',12, ...
    'Box','on')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'contrast_scale_098_100.pdf'), ...
    'ContentType','vector', ...
    'BackgroundColor','white')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'contrast_scale_098_100.png'), ...
    'Resolution',300, ...
    'BackgroundColor','white')

close(fig)


%% ================================================================
%  AXISYMMETRIC PHYSICAL VELOCITY STREAMLINES
%
%  FEniCS writes:
%
%       u_physical.xdmf
%       u_physical.h5
%
%  The stored field is the physical meridional velocity
%
%       (u_r,u_z)
%
%  on the positive-r half-domain.
%
%  For a full meridional plot:
%
%       u_x(-r,z) = -u_r(r,z)
%       u_z(-r,z) =  u_z(r,z)
%
%  so the radial component changes sign when reflected across the
%  symmetry axis.
%
%  Clean streamline figures at 200 and 500 s are written directly
%  to the THESIS_READY folder.  If 500 s is not yet available, it is
%  simply skipped until the script is rerun later.


if ...
    isfile(velocityXdmfFile) ...
    && ...
    isfile(velocityH5File)

    fprintf('\nReading physical velocity output...\n')


    velocityXml = fileread(velocityXdmfFile);

    velocityTokens = regexp( ...
        velocityXml, ...
        '<Time\s+Value="([^"]+)"', ...
        'tokens');


    nVelocityTimes = numel(velocityTokens);


    if nVelocityTimes == 0

        warning( ...
            'No saved velocity times were found in %s.', ...
            velocityXdmfFile)

    else

        velocityTimes = zeros(nVelocityTimes,1);


        for kk = 1:nVelocityTimes

            velocityTimes(kk) = ...
                str2double(velocityTokens{kk}{1});

        end


        streamlineTimesUse = streamlineTimes( ...
            streamlineTimes >= velocityTimes(1) ...
            & ...
            streamlineTimes <= velocityTimes(end));


        fprintf( ...
            'Physical velocity available from %.3f to %.3f s.\n', ...
            velocityTimes(1), ...
            velocityTimes(end))


        if isempty(streamlineTimesUse)

            warning( ...
                ['None of the requested streamline times are yet ' ...
                 'available. Requested: %s'], ...
                mat2str(streamlineTimes))

        else

            % Physical full-domain plotting grid.
            rStream = linspace(-R,R,nStreamR);
            zStream = linspace(0,H,nStreamZ);

            [Rstream,Zstream] = meshgrid( ...
                rStream, ...
                zStream);

            Sstream = 0.5*abs(Rstream).^2;


            for jj = 1:numel(streamlineTimesUse)

                requestedTime = streamlineTimesUse(jj);

                [~,velocityIndex] = min( ...
                    abs(velocityTimes-requestedTime));

                tVelocity = velocityTimes(velocityIndex);


                velocityDataset = sprintf( ...
                    '/VisualisationVector/%d', ...
                    velocityIndex-1);


                rawVelocity = h5read( ...
                    velocityH5File, ...
                    velocityDataset);


                [urNode,uzNode] = ...
                    unpackFenicsVelocityForMatlab( ...
                        rawVelocity, ...
                        nNodes, ...
                        nCells);


                % Interpolate the positive-r physical velocity in
                % transformed coordinates (s,z).
                Fur = scatteredInterpolant( ...
                    sNode, ...
                    zNode, ...
                    urNode, ...
                    'linear', ...
                    'nearest');


                Fuz = scatteredInterpolant( ...
                    sNode, ...
                    zNode, ...
                    uzNode, ...
                    'linear', ...
                    'nearest');


                urPositive = Fur( ...
                    Sstream, ...
                    Zstream);


                uzFull = Fuz( ...
                    Sstream, ...
                    Zstream);


                % Correct reflection:
                %
                % positive physical u_r points to increasing r.
                % On the negative half of the mirrored meridional
                % plane the horizontal component therefore reverses.
                uxFull = sign(Rstream).*urPositive;


                % Enforce symmetry exactly on the centreline.
                uxFull(abs(Rstream) < 1e-12) = 0;


                % Remove any non-finite interpolation values.
                uxFull(~isfinite(uxFull)) = 0;
                uzFull(~isfinite(uzFull)) = 0;



                % CLEAN THESIS STREAMLINE FIGURE


                fig = figure( ...
                    'Color','w', ...
                    'Visible','off', ...
                    'Units','pixels', ...
                    'Position',[100 100 1800 900]);


                ax = axes( ...
                    'Parent',fig, ...
                    'Units','normalized', ...
                    'Position',[0 0 1 1]);


                hold(ax,'on')


                hStream = streamslice( ...
                    ax, ...
                    Rstream, ...
                    Zstream, ...
                    uxFull, ...
                    uzFull, ...
                    streamlineDensity);


                % Dark red streamlines, matching the numerical
                % profile colour family used elsewhere.
                set( ...
                    hStream, ...
                    'Color',[0.55 0 0], ...
                    'LineWidth',0.9)


                % Domain boundary.
                rectangle( ...
                    ax, ...
                    'Position',[-R 0 2*R H], ...
                    'EdgeColor',[0 0 0], ...
                    'LineWidth',1.2)


                xlim(ax,[-R R])
                ylim(ax,[0 H])

                axis(ax,'equal')
                axis(ax,'off')


                timeTagVelocity = sprintf( ...
                    '%07.3fs', ...
                    tVelocity);

                timeTagVelocity = strrep( ...
                    timeTagVelocity, ...
                    '.', ...
                    'p');


                exportgraphics( ...
                    ax, ...
                    fullfile( ...
                        streamlineFolder, ...
                        ['streamlines_' timeTagVelocity '.pdf']), ...
                    'ContentType','vector', ...
                    'BackgroundColor','white')


                exportgraphics( ...
                    ax, ...
                    fullfile( ...
                        streamlineFolder, ...
                        ['streamlines_' timeTagVelocity '.png']), ...
                    'Resolution',600, ...
                    'BackgroundColor','white')


                % Convenient thesis-ready copies.
                exportgraphics( ...
                    ax, ...
                    fullfile( ...
                        thesisReadyFolder, ...
                        ['STREAMLINES_' timeTagVelocity '.pdf']), ...
                    'ContentType','vector', ...
                    'BackgroundColor','white')


                exportgraphics( ...
                    ax, ...
                    fullfile( ...
                        thesisReadyFolder, ...
                        ['STREAMLINES_' timeTagVelocity '.png']), ...
                    'Resolution',600, ...
                    'BackgroundColor','white')


                close(fig)


                fprintf( ...
                    'Saved streamline field at %.3f s.\n', ...
                    tVelocity)

            end

        end

    end

else

    warning( ...
        ['Velocity streamline files were not found. Concentration ' ...
         'post-processing will still finish. Expected:\n  %s\n  %s'], ...
        velocityXdmfFile, ...
        velocityH5File)

end


%% ================================================================
%  BROADER THESIS CONTRAST SCALE REFERENCE: 0.97--1


fig = figure( ...
    'Color','w', ...
    'Visible','off', ...
    'Units','pixels', ...
    'Position',[100 100 800 180]);

ax = axes( ...
    'Parent',fig, ...
    'Position',[0.08 0.55 0.84 0.18]);

imagesc( ...
    ax, ...
    linspace(0.97,1,512))

set(ax,'YTick',[])
clim(ax,[0.97 1])
colormap(ax,flipud(gray(256)))

xlabel( ...
    ax, ...
    'Concentration, $c$', ...
    'Interpreter','latex')

set(ax, ...
    'FontSize',12, ...
    'Box','on')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'contrast_scale_097_100.pdf'), ...
    'ContentType','vector', ...
    'BackgroundColor','white')

exportgraphics( ...
    fig, ...
    fullfile( ...
        analysisFolder, ...
        'contrast_scale_097_100.png'), ...
    'Resolution',300, ...
    'BackgroundColor','white')

close(fig)


%% ================================================================
%  SAVE NUMERICAL DATA


save( ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_data.mat'), ...
    'times', ...
    'zAverage', ...
    'cBarAll', ...
    'cBarNormalisedSelected', ...
    'cVolumeMean', ...
    'freshVolumeFraction', ...
    'profileTimes', ...
    'thesisFieldTimes', ...
    'R', ...
    'H', ...
    'broadThesisContrastFreshMax');


%% Long-form CSV:
%
%      time_s    z_over_H    c_mean

timeColumn = repelem( ...
    times, ...
    nProfileBins);


heightColumn = repmat( ...
    zAverage/H, ...
    nTimes, ...
    1);


concentrationColumn = cBarAll(:);


outputTable = table( ...
    timeColumn, ...
    heightColumn, ...
    concentrationColumn, ...
    'VariableNames', ...
    {'time_s','z_over_H','c_mean'});


writetable( ...
    outputTable, ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_average_all_data.csv'));


% Compact time-series summary
summaryTable = table( ...
    times, ...
    cVolumeMean, ...
    freshVolumeFraction, ...
    'VariableNames', ...
    {'time_s','volume_mean_concentration','volume_mean_fresh_fraction'});

writetable( ...
    summaryTable, ...
    fullfile( ...
        resultsFolder, ...
        'axisymmetric_time_summary.csv'));


%% ================================================================
%  FINISHED


fprintf('\n============================================\n')
fprintf('Finished.\n')
fprintf('Processed %d DG0 concentration fields.\n',nTimes)
fprintf('Mesh vertices = %d\n',nNodes)
fprintf('Mesh cells    = %d\n',nCells)
fprintf('Saved ALL %d raw concentration snapshots.\n',numel(fieldIdx))
fprintf('Saved ALL %d contrast-enhanced snapshots.\n',numel(fieldIdx))
fprintf('Last exported field time = %.6g s\n',times(fieldIdx(end)))
fprintf('MAIN thesis contrast: 0.98 <= c <= 1\n')
fprintf('Diagnostic contrast: 0.99 <= c <= 1\n')
fprintf('Broader thesis contrast: 0.97 <= c <= 1\n')
fprintf('Clean thesis-ready concentration/profile/streamline outputs saved in:\n')
fprintf('    %s\n',thesisReadyFolder)
fprintf('Selected thesis fields: %d\n',numel(thesisFieldIdx))
fprintf('Saved %d selected mean profiles.\n',numel(profileIdx))
fprintf('Results saved in:\n')
fprintf('    %s\n',resultsFolder)
fprintf('============================================\n')


%% ================================================================
%  LOCAL FUNCTION: UNPACK FEniCS PHYSICAL VELOCITY
%
%  XDMF/HDF5 vector output can appear in MATLAB as either:
%
%      N x 2, 2 x N,
%      N x 3, 3 x N,
%
%  depending on HDF5 dimension ordering and whether FEniCS pads the
%  visualisation vector to three components.
%
%  The final axisymmetric physical velocity should be NODE based in
%  the visualisation output.  A cell-based fallback is included for
%  robustness.


function [ur,uz] = unpackFenicsVelocityForMatlab( ...
    rawVelocity, ...
    nNodes, ...
    nCells)

    rawVelocity = double(rawVelocity);

    sz = size(rawVelocity);



    % NODES x COMPONENTS


    if ...
        numel(sz) >= 2 ...
        && ...
        sz(1) == nNodes ...
        && ...
        (sz(2) == 2 || sz(2) == 3)

        ur = rawVelocity(:,1);
        uz = rawVelocity(:,2);
        return

    end



    % COMPONENTS x NODES


    if ...
        numel(sz) >= 2 ...
        && ...
        sz(2) == nNodes ...
        && ...
        (sz(1) == 2 || sz(1) == 3)

        ur = rawVelocity(1,:).';
        uz = rawVelocity(2,:).';
        return

    end



    % FLAT NODE VECTOR
    %
    % Try 2-component and 3-component interleaved layouts.


    rawFlat = rawVelocity(:);


    if numel(rawFlat) == 2*nNodes

        tmp = reshape( ...
            rawFlat, ...
            2, ...
            nNodes);

        ur = tmp(1,:).';
        uz = tmp(2,:).';
        return

    end


    if numel(rawFlat) == 3*nNodes

        tmp = reshape( ...
            rawFlat, ...
            3, ...
            nNodes);

        ur = tmp(1,:).';
        uz = tmp(2,:).';
        return

    end



    % If we get here, the visualisation file is not nodal in a form
    % that can be reconstructed from the original mesh vertices.


    error( ...
        ['Could not interpret velocity dataset dimensions [%s]. ' ...
         'Expected 2 or 3 components for %d mesh vertices. ' ...
         'Raw dataset contains %d values.'], ...
        num2str(sz), ...
        nNodes, ...
        numel(rawFlat))

end


%% ================================================================
%  LOCAL FUNCTION: SAVE EXACT PORTRAIT FIGURE
%
%  exportgraphics can tightly crop axis-off figures.  print() uses an
%  explicitly specified paper size instead, so the saved PNG/PDF keep
%  the required 5:8 thesis aspect ratio.


function saveExactPortraitFigure( ...
    fig, ...
    pngFile, ...
    pdfFile, ...
    paperSizeInches, ...
    dpi)

    widthIn = paperSizeInches(1);
    heightIn = paperSizeInches(2);

    set(fig, ...
        'PaperUnits','inches', ...
        'PaperPosition',[0 0 widthIn heightIn], ...
        'PaperSize',[widthIn heightIn], ...
        'PaperPositionMode','manual', ...
        'InvertHardcopy','off');

    drawnow

    print( ...
        fig, ...
        pngFile, ...
        '-dpng', ...
        sprintf('-r%d',dpi));

    print( ...
        fig, ...
        pdfFile, ...
        '-dpdf', ...
        '-painters');

end
