import unittest

from foodverse_core import (
    build_weekly_plan,
    calculate_targets,
    demo_food_analysis,
    demo_freshness,
    demo_kitchen_analysis,
    normalize_food_data,
    normalize_freshness_data,
    normalize_kitchen_data,
)


class FoodVerseCoreTests(unittest.TestCase):
    def test_food_demo_has_nonnegative_nutrients_and_score(self):
        result = normalize_food_data(demo_food_analysis("Dal and rice"))
        self.assertGreaterEqual(result["nutrition"]["calories_kcal"], 0)
        self.assertGreaterEqual(result["health_score"], 1)
        self.assertLessEqual(result["health_score"], 100)

    def test_freshness_action_is_conservative_for_spoiled(self):
        result = normalize_freshness_data(demo_freshness("Visible mold / sliminess", "Fruit"), "Fruit")
        self.assertEqual(result["freshness_stage"], "Spoiled")
        self.assertEqual(result["action"], "Discard immediately")

    def test_kitchen_score_is_clamped(self):
        result = normalize_kitchen_data({"compliance_score": 150, "checks": []})
        self.assertEqual(result["compliance_score"], 100)

    def test_week_has_seven_days_and_allocation_adds_up(self):
        targets = calculate_targets(30, "Female", 70, 165, "Moderately active", "Fat Loss")
        plan, grocery = build_weekly_plan("Vegan", int(targets["target_kcal"]))
        self.assertEqual(len(plan), 7)
        self.assertTrue(grocery)
        for day in plan:
            self.assertEqual(sum(meal["target_kcal"] for meal in day["meals"]), targets["target_kcal"])

    def test_bmr_requires_adult(self):
        with self.assertRaises(ValueError):
            calculate_targets(17, "Female", 55, 160, "Sedentary", "Maintain Weight")


if __name__ == "__main__":
    unittest.main()
