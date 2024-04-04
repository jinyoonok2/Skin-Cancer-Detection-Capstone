from _yolo.evaluation import ModelEvaluation

def main(setting_number=1):
    conf_scores = [0.25, 0.35, 0.50, 0.65]

    model_evaluator = ModelEvaluation(setting_number)
    model_evaluator.evaluate(conf_scores)
    model_evaluator.yolo_builtin_evaluation()

if __name__ == "__main__":
    main(3)
