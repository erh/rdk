package armplanning

import (
	"context"
	"testing"

	"github.com/golang/geo/r3"
	"go.viam.com/test"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	frame "go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
)

func TestSmoothVelocityXArm6Linear(t *testing.T) {
	logger := logging.NewTestLogger(t)

	model, err := frame.ParseModelJSONFile(utils.ResolveFile("components/arm/fake/kinematics/xarm6.json"), "")
	test.That(t, err, test.ShouldBeNil)

	fs := frame.NewEmptyFrameSystem("")
	err = fs.AddFrame(model, fs.World())
	test.That(t, err, test.ShouldBeNil)

	startInputs := frame.NewZeroInputs(fs)

	// xarm6 at zero joints is at approximately (207, 0, 112) with OZ: -1
	// move 100mm in X from there
	goal := spatialmath.NewPose(r3.Vector{X: 307, Y: 0, Z: 112}, &spatialmath.OrientationVector{OZ: -1})

	constraints := &motionplan.Constraints{
		LinearConstraint: []motionplan.LinearConstraint{
			{LineToleranceMm: 1, OrientationToleranceDegs: 1},
		},
	}

	req := &PlanRequest{
		FrameSystem: fs,
		Goals: []*PlanState{
			{poses: frame.FrameSystemPoses{model.Name(): frame.NewPoseInFrame(frame.World, goal)}},
		},
		StartState:     &PlanState{structuredConfiguration: startInputs},
		Constraints:    constraints,
		PlannerOptions: NewBasicPlannerOptions(),
	}

	plan, _, err := PlanMotion(context.Background(), logger, req)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(plan.Trajectory()), test.ShouldBeGreaterThan, 40)
	test.That(t, len(plan.Trajectory()), test.ShouldBeLessThan, 60)

	stats := TrajectoryDeltaStats(plan.Trajectory())
	for _, s := range stats {
		test.That(t, s.StdDev, test.ShouldBeLessThan, .00001)
	}
}
