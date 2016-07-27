require "spec"
require "./ml"
require "./trees"

# examples comming from: http://www.saedsayad.com/decision_tree.htm
describe ML do
  describe "entropy" do
    it "should calculate entropy and gain given categorical features" do
      f1 = %i(r r o s s s o r r s r o o s)  # outlook
      f2 = %i(h h h m c c c m c m m m h m)  # temperature
      f3 = %i(h h h h n n n h n n n h n h)  # humidity
      f4 = %i(f t f f f t t f f f t t f t)  # windy
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      ML.entropy(y).should be_close(0.94, delta: 0.01)

      ML.entropy(y, given: f1).should be_close(0.693, delta: 0.01)

      ML.gain(y, given: f1).should be_close(0.247, delta: 0.01)
      ML.gain(y, given: f2).should be_close(0.029, delta: 0.01)
      ML.gain(y, given: f3).should be_close(0.152, delta: 0.01)
      ML.gain(y, given: f4).should be_close(0.048, delta: 0.01)

    end

    it "should calculate std given categorical features and continuous output" do
      f1 = %i(r r o s s s o r r s r o o s)  # outlook
      f2 = %i(h h h m c c c m c m m m h m)  # temperature
      f3 = %i(h h h h n n n h n n n h n h)  # humidity
      f4 = %i(f t f f f t t f f f t t f t)  # windy
      y =  [25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30]  # hours played

      y.std.should be_close(9.32, delta: 0.01)
      ML.std(y, given: f1).should be_close(7.66, delta: 0.01)
    end



  end
end
