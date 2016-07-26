require "spec"
require "./ml"
require "./trees"

describe ML do
  describe "entropy" do
    it "should calculate tags entropy" do
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      ML.entropy(tags: y).should be_close(0.94, delta: 0.01)
    end

    it "should calculate tags entropy given categorical feature" do
      f1 = %i(r r o s s s o r r s r o o s)  # outlook
      f2 = %i(h h h m c c c m c m m m h m)  # temperature
      f3 = %i(h h h h n n n h n n n h n h)  # humidity
      f4 = %i(f t f f f t t f f f t t f t)  # windy
      y =  %i(n n y y y n y n y y y y y n)  # play golf?

      ML.entropy(tags: y, given: f1).should be_close(0.693, delta: 0.01)

      ML.gain(tags: y, given: f1).should be_close(0.247, delta: 0.01)
      ML.gain(tags: y, given: f2).should be_close(0.029, delta: 0.01)
      ML.gain(tags: y, given: f3).should be_close(0.152, delta: 0.01)
      ML.gain(tags: y, given: f4).should be_close(0.048, delta: 0.01)

    end

  end
end
