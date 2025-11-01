import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Activity, AlertCircle, CheckCircle2, Info } from "lucide-react";
import { toast } from "sonner";

export default function PCOSAssessment() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<boolean[]>([]);
  const [showResults, setShowResults] = useState(false);

  const questions = [
    "Do you experience irregular periods (cycles longer than 35 days or shorter than 21 days)?",
    "Do you have excessive hair growth on face, chest, or back?",
    "Have you noticed hair thinning or hair loss on your scalp?",
    "Do you struggle with acne or oily skin?",
    "Have you experienced unexplained weight gain or difficulty losing weight?",
    "Do you have darkened skin patches, especially around neck or armpits?",
    "Have you been diagnosed with insulin resistance or pre-diabetes?",
    "Do you experience pelvic pain or discomfort?",
  ];

  const handleAnswer = (answer: boolean) => {
    const newAnswers = [...answers, answer];
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      setShowResults(true);
      toast.success("Assessment completed");
    }
  };

  const resetAssessment = () => {
    setCurrentQuestion(0);
    setAnswers([]);
    setShowResults(false);
  };

  const riskScore = answers.filter((a) => a).length;
  const getRiskLevel = () => {
    if (riskScore <= 2) return { level: "Low", color: "text-secondary", bg: "bg-secondary/20" };
    if (riskScore <= 4) return { level: "Moderate", color: "text-primary", bg: "bg-primary/20" };
    return { level: "High", color: "text-destructive", bg: "bg-destructive/20" };
  };

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  return (
    <div className="min-h-screen bg-[var(--gradient-subtle)]">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl">
            <Activity className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">PCOS Assessment</h1>
            <p className="text-muted-foreground">Screening tool for PCOS symptoms</p>
          </div>
        </div>

        {/* Info Card */}
        <Card className="p-6 bg-gradient-to-br from-accent/10 to-accent/5 border-accent/20">
          <div className="flex gap-3">
            <Info className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-foreground mb-2">About PCOS</h3>
              <p className="text-sm text-muted-foreground">
                Polycystic Ovary Syndrome (PCOS) is a hormonal disorder affecting women of
                reproductive age. This assessment helps identify potential symptoms but is not a
                diagnosis. Please consult a healthcare provider for proper evaluation.
              </p>
            </div>
          </div>
        </Card>

        {!showResults ? (
          /* Assessment Questions */
          <Card className="p-8">
            <div className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>Question {currentQuestion + 1} of {questions.length}</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>

              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-foreground leading-relaxed">
                  {questions[currentQuestion]}
                </h2>

                <div className="flex gap-4">
                  <Button
                    onClick={() => handleAnswer(true)}
                    className="flex-1 h-16 bg-gradient-to-br from-primary to-primary/80 hover:opacity-90"
                  >
                    Yes
                  </Button>
                  <Button
                    onClick={() => handleAnswer(false)}
                    variant="outline"
                    className="flex-1 h-16 hover:border-primary"
                  >
                    No
                  </Button>
                </div>
              </div>

              {currentQuestion > 0 && (
                <Button
                  onClick={() => {
                    setCurrentQuestion(currentQuestion - 1);
                    setAnswers(answers.slice(0, -1));
                  }}
                  variant="ghost"
                  className="w-full"
                >
                  Previous Question
                </Button>
              )}
            </div>
          </Card>
        ) : (
          /* Results */
          <div className="space-y-6">
            <Card className="p-8">
              <div className="text-center space-y-4">
                <div className={`inline-flex p-4 rounded-full ${getRiskLevel().bg}`}>
                  {getRiskLevel().level === "Low" ? (
                    <CheckCircle2 className={`w-12 h-12 ${getRiskLevel().color}`} />
                  ) : (
                    <AlertCircle className={`w-12 h-12 ${getRiskLevel().color}`} />
                  )}
                </div>

                <div>
                  <h2 className="text-2xl font-bold text-foreground mb-2">Assessment Complete</h2>
                  <p className="text-muted-foreground">
                    You answered "Yes" to {riskScore} out of {questions.length} questions
                  </p>
                </div>

                <Badge className={`${getRiskLevel().bg} ${getRiskLevel().color} text-lg px-6 py-2`}>
                  {getRiskLevel().level} Risk Level
                </Badge>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Recommendations</h3>
              <div className="space-y-3">
                {getRiskLevel().level !== "Low" && (
                  <div className="flex gap-3 p-4 rounded-lg bg-primary/10">
                    <AlertCircle className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-foreground">Consult a Healthcare Provider</p>
                      <p className="text-sm text-muted-foreground">
                        Consider speaking with a gynecologist or endocrinologist for proper diagnosis
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex gap-3 p-4 rounded-lg bg-accent/10">
                  <CheckCircle2 className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground">Lifestyle Management</p>
                    <p className="text-sm text-muted-foreground">
                      Regular exercise, balanced diet, and stress management can help
                    </p>
                  </div>
                </div>

                <div className="flex gap-3 p-4 rounded-lg bg-secondary/10">
                  <CheckCircle2 className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground">Track Your Symptoms</p>
                    <p className="text-sm text-muted-foreground">
                      Use RituCare to monitor your cycle and symptoms over time
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Button onClick={resetAssessment} variant="outline" className="w-full">
              Retake Assessment
            </Button>
          </div>
        )}

        {/* Educational Content */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Understanding PCOS</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-medium text-foreground">Common Symptoms</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Irregular menstrual cycles</li>
                <li>• Excessive hair growth</li>
                <li>• Acne and oily skin</li>
                <li>• Weight gain</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-foreground">Management Options</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Lifestyle modifications</li>
                <li>• Medications (if prescribed)</li>
                <li>• Regular monitoring</li>
                <li>• Dietary changes</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
