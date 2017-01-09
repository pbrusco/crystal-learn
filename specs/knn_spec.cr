require "spec"
require "csv"
require "../lib/ml"
require "../lib/knn"
require "../lib/array"
require "../lib/random"
require "../lib/math"

describe "with numerical data (iris dataset)" do
  it "should classify" do
    x, y = ML.load_floats_csv("iris.csv")
    clf = ML::Classifiers::KNeighborsClassifier(Float32, String).new.fit(x, y)
    y_pred = clf.predict(x)

    acc = ML.accuracy(y, y_pred)
    acc.is_a?(Float).should eq(true)
    acc.round(2).should eq(0.97)
  end
end
