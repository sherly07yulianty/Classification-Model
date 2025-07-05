## Import Packages and Libraries
using DataFrames
using DelimitedFiles
using CSV
using XLSX
using RDatasets
using PrettyPrinting #menampilkan skema data yang lebih menarik
using DataFrames 
using MLJ
using ScientificTypes #mengidentifikasi typenya, misal numerik dll
using StableRNGs #random number generator
using Statistics
using Plots
using StatsBase
using StatsPlots
using Distributions
using LinearAlgebra
using HypothesisTests
using GLMNet
using RDatasets
using DecisionTree
using Distances
using NearestNeighbors
using Random
using DataStructures
using LIBSVM
using Imbalance

## Import and Choosing Data
C1 = CSV.read("diabetes.csv", DataFrame);
coerce!(C1, :Outcome => Multiclass)
schema(C1)
select!(C1, Not([:BloodPressure, :SkinThickness, :DiabetesPedigreeFunction]))
y, X = unpack(C1, ==(:Outcome), colname -> true) # pisahkan data respon dan prediktor
first(X, 3) |> pretty

## Display a list of models that match the dataset
for m in models(matching(X, y)) 
    println(rpad(m.name, 40), "($(m.package_name))") 
end

## Check for missing data in each column
missing_per_column = map(col -> any(ismissing, col), eachcol(C1))
println("\nMissing data di setiap kolom:")
println(missing_per_column)

## Looking at the proportion of sick vs non-sick data
count_sick = count(x -> x == 1, C1.Outcome)
count_health = count(x -> x == 0, C1.Outcome)
values = [count_sick, count_health]
colors = [:blue, :orange]

# Calculate percentages and merge with labels
total = sum(values)
labels_with_percentages = ["Diabetes: $(round(100 * values[1] / total, digits=1))%", 
                           "Not Diabetes: $(round(100 * values[2] / total, digits=1))%"]

# Creating a pie chart without labels outside the chart
pie_chart = pie(values, label = "", colors = colors, title = "Proportion of Sick vs Not Sick", legend = :outertopright)

# Add labels and percentages to each section of the pie chart with white text
for i in 1:length(values)
    angle = sum(values[1:i]) - values[i] / 2  
    x = cos(angle * 2 * π / total) * 0.5      
    y = sin(angle * 2 * π / total) * 0.5      
    annotate!(x, y, text(labels_with_percentages[i], :center, 10, :white))
end
# Displaying a pie chart
pie_chart

X = Matrix(X)
xunder, yunder = random_oversample(X, y, ratios=Dict(0=>1.0, 1=> 0.7))

## Create a pie chart for balanced data
count_sick = count(xunder -> xunder == 1, yunder)
count_health = count(xunder -> xunder == 0, yunder)
values = [count_sick, count_health]
colors = [:blue, :orange]

# Calculate percentages and merge with labels
total = sum(values)
labels_with_percentages = ["Diabetes: $(round(100 * values[1] / total, digits=1))%", 
                           "Not Diabetes: $(round(100 * values[2] / total, digits=1))%"]

# Creating a pie chart without labels outside the chart
pie_chart = pie(values, label = "", colors = colors, title = "Proportion of Sick vs Not Sick", legend = :outertopright)

# Add labels and percentages to each section of the pie chart with white text
for i in 1:length(values)
    angle = sum(values[1:i]) - values[i] / 2  
    x = cos(angle * 2 * π / total) * 0.5      
    y = sin(angle * 2 * π / total) * 0.5      
    annotate!(x, y, text(labels_with_percentages[i], :center, 10, :white))
end

# Displaying a pie chart
pie_chart

## Dividing the dataset into training and testing
function perclass_splits(y,at)
    uids = unique(y)
    keepids = []
    for ui in uids
        curids = findall(y.==ui)
        rowids = randsubseq(curids, at) 
        push!(keepids,rowids...)
    end
    return keepids
end

trainids = perclass_splits(yunder,0.8)
testids = setdiff(1:length(yunder),trainids)

## Measuring model accuracy
findaccuracy(predictedvals,groundtruthvals) = sum(predictedvals.==groundtruthvals)/length(groundtruthvals)

#Decision Tree
@time begin
    model1 = DecisionTreeClassifier(max_depth=25)
    DecisionTree.fit!(model1, xunder[trainids,:], yunder[trainids])
end
q1 = xunder[testids,:];
predictions_DT = DecisionTree.predict(model1, q1)
findaccuracy(predictions_DT,yunder[testids])

#Random Forest
@time begin
    model2 = RandomForestClassifier(n_trees=43)
    DecisionTree.fit!(model2, xunder[trainids,:], yunder[trainids])
end
q2 = xunder[testids,:];
predictions_RF = DecisionTree.predict(model2, q2)
findaccuracy(predictions_RF,yunder[testids])

#Nearest Neighbor
@time begin
    Xtrain = xunder[trainids,:]
    ytrain = yunder[trainids]
    kdtree = KDTree(Xtrain')
end
queries = xunder[testids,:]
idxs, dists = knn(kdtree, queries', 23, true)
c = ytrain[hcat(idxs...)]
possible_labels = map(i->counter(c[:,i]),1:size(c,2))
predictions_NN = map(i->parse(Int,string(string(argmax(possible_labels[i])))),1:size(c,2))
findaccuracy(predictions_NN,yunder[testids])

## Normalization 
y, X = unpack(C1, ==(:Outcome), colname -> true) 
first(X, 3) |> pretty
# Min-Max Normalization function for each column
function min_max_normalize(X)
    return DataFrame([ (X[!, col] .- minimum(X[!, col])) ./ (maximum(X[!, col]) - minimum(X[!, col])) for col in names(X) ], names(X))
end
normalized_X = min_max_normalize(X)
println(normalized_X)

X1 = Matrix(normalized_X)
xunder1, yunder1 = random_oversample(X1, y, ratios=Dict(0=>1.0, 1=> 0.7))

## Dividing the dataset into training and testing
function perclass_splits(y,at)
    uids = unique(y)
    keepids = []
    for ui in uids
        curids = findall(y.==ui)
        rowids = randsubseq(curids, at) 
        push!(keepids,rowids...)
    end
    return keepids
end
trainids1 = perclass_splits(yunder1,0.8)
testids1 = setdiff(1:length(yunder1),trainids1)

#Decision Tree
@time begin
    model4 = DecisionTreeClassifier(max_depth=25)
    DecisionTree.fit!(model4, xunder1[trainids1,:], yunder1[trainids1])
end
q4 = xunder1[testids1,:];
predictions_DT1 = DecisionTree.predict(model4, q4)
findaccuracy(predictions_DT1,yunder1[testids1])

#Random Forest
@time begin
    model5 = RandomForestClassifier(n_trees=43)
    DecisionTree.fit!(model5, xunder1[trainids1,:], yunder1[trainids1])
end
q5 = xunder1[testids1,:];
predictions_RF1 = DecisionTree.predict(model5, q5)
findaccuracy(predictions_RF1,yunder1[testids1])

#Nearest Neighbor
@time begin
    Xtrain1 = xunder1[trainids1,:]
    ytrain1 = yunder1[trainids1]
    kdtree1 = KDTree(Xtrain1')
end
queries1 = xunder1[testids1,:]
idxs1, dists1 = knn(kdtree1, queries1', 23, true)
c1 = ytrain1[hcat(idxs1...)]
possible_labels1 = map(i->counter(c1[:,i]),1:size(c1,2))
predictions_NN1 = map(i->parse(Int,string(string(argmax(possible_labels1[i])))),1:size(c1,2))
findaccuracy(predictions_NN1,yunder1[testids1])
