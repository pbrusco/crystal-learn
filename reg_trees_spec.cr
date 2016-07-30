require "spec"
require "csv"
require "./ml"
require "./trees"
require "./array"

describe ML::Classifiers::DecisionTreeRegresor do
  describe "with categorical data (hair eye color)" do
    it "with training data" do
      x, y = ML.load_string_csv("HairEyeColor.csv", columns_to_skip: 1)
      clf = ML::Classifiers::DecisionTreeRegresor.new.fit(x, y)
      y_pred = clf.predict(x)
      y_pred.should eq([32, 53, 10, 3.5, 11, 50, 10, 30, 10, 25, 7, 5, 2.5, 14.5, 7, 8, 36, 66, 16, 3.5, 9, 34, 7, 64, 5, 29, 7, 5, 2.5, 14.5, 7, 8])
    end
  end

  it "with unseen data" do
    x, y = ML.load_string_csv("HairEyeColor.csv", columns_to_skip: 1)
    train_index, test_index = ML.train_test_split(x.size, train_size: 0.8)

    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    clf = ML::Classifiers::DecisionTreeRegresor.new.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    acc = ML.accuracy(y_test, y_pred)
    acc.should eq(0.5)
  end
end
