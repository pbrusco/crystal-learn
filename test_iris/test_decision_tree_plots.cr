require "csv"
require "../ml"
require "../trees"

pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

iris_data, iris_target = ML.load_floats_csv("../iris.csv")

pairs.each_with_index do |pair, pairidx|
  # We only take the two corresponding features
  x = iris_data.keep_columns(pair)
  y = iris_target

  # Standardize
  x = ML.standardize(x)

  # Train
  clf = ML::Classifiers::DecisionTreeClassifier.new.fit(x, y)

  f = File.open("csv_for_pair_#{pairidx}.csv", mode: "w")

  result = CSV.build(f) do |csv|
    ML.arange(-4.0, 4, step: 0.2) do |f2|
      ML.arange(-4.0, 4, step: 0.2) do |f1|
        z = clf.predict([f1, f2])
        csv.row f1, f2, z, "False"
      end
    end
    x.each_with_index do |row, idx|
      csv.row row[0], row[1], y[idx], "True"
    end
  end

  f.close()


end
