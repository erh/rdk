package armplanning

import (
	"context"
	"fmt"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/referenceframe"
)

func smoothVelocities(ctx context.Context,
	segmentContexts []*planSegmentContext,
	traj []*referenceframe.LinearInputs,
	logger logging.Logger,
) ([]*referenceframe.LinearInputs, error) {
	if len(segmentContexts)+1 != len(traj) {
		// because we put the start position
		return nil, fmt.Errorf("lengths wrong segmentContexts: %d linearinputs: %d", len(segmentContexts), len(traj))
	}

	fs := segmentContexts[0].pc.fs
	start := traj[0]
	end := traj[len(traj)-1]
	lastIdx := len(traj) - 1

	// Build a new trajectory using straight-line joint interpolation between
	// start and end. This produces perfectly constant joint deltas.
	newTraj := make([]*referenceframe.LinearInputs, len(traj))
	newTraj[0] = start
	newTraj[lastIdx] = end

	replaced := 0

	for i := 1; i < lastIdx; i++ {
		by := float64(i) / float64(lastIdx)

		interpJoints, err := referenceframe.InterpolateFS(fs, start, end, by)
		if err != nil {
			return traj, fmt.Errorf("interpolation error at step %d: %w", i, err)
		}

		logger.Infof("eliot %d %v", i, logging.FloatArrayFormat{"%0.5f", interpJoints.GetLinearizedInputs()})

		// Pick the segment context for constraint checking
		pscIdx := i
		if pscIdx >= len(segmentContexts) {
			pscIdx = len(segmentContexts) - 1
		}
		psc := segmentContexts[pscIdx]

		// Check constraints on the interpolated joints
		_, constraintErr := psc.checker.CheckStateFSConstraints(ctx, &motionplan.StateFS{
			Configuration: interpJoints,
			FS:            fs,
		})

		if constraintErr == nil {
			newTraj[i] = interpJoints
			replaced++
		} else {
			logger.Infof("smoothVelocities step %d/%d interp fails constraints: %v", i, lastIdx, constraintErr)
			newTraj[i] = traj[i]
		}
	}

	logger.Infof("smoothVelocities replaced %d/%d intermediate trajectory points", replaced, lastIdx-1)
	return newTraj, nil
}
