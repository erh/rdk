// Package slam implements simultaneous localization and mapping
// This is an Experimental package
package slam

import (
	"context"
	"image"
	"io"
	"sync"

	"github.com/edaniels/golog"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
	pb "go.viam.com/api/service/slam/v1"
	goutils "go.viam.com/utils"
	"go.viam.com/utils/rpc"

	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/registry"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/subtype"
	"go.viam.com/rdk/utils"
	"go.viam.com/rdk/vision"
)

// TBD 05/04/2022: Needs more work once GRPC is included (future PR).
func init() {
	registry.RegisterResourceSubtype(Subtype, registry.ResourceSubtype{
		RegisterRPCService: func(ctx context.Context, rpcServer rpc.Server, subtypeSvc subtype.Service) error {
			return rpcServer.RegisterServiceServer(
				ctx,
				&pb.SLAMService_ServiceDesc,
				NewServer(subtypeSvc),
				pb.RegisterSLAMServiceHandlerFromEndpoint,
			)
		},
		RPCServiceDesc: &pb.SLAMService_ServiceDesc,
		RPCClient: func(ctx context.Context, conn rpc.ClientConn, name string, logger golog.Logger) interface{} {
			return NewClientFromConn(ctx, conn, name, logger)
		},
		Reconfigurable: WrapWithReconfigurable,
	})
}

// NewUnimplementedInterfaceError is used when there is a failed interface check.
func NewUnimplementedInterfaceError(actual interface{}) error {
	return utils.NewUnimplementedInterfaceError((*Service)(nil), actual)
}

// SubtypeName is the name of the type of service.
const SubtypeName = resource.SubtypeName("slam")

// Subtype is a constant that identifies the slam resource subtype.
var Subtype = resource.NewSubtype(
	resource.ResourceNamespaceRDK,
	resource.ResourceTypeService,
	SubtypeName,
)

// Named is a helper for getting the named service's typed resource name.
func Named(name string) resource.Name {
	return resource.NameFromSubtype(Subtype, name)
}

var (
	_ = Service(&reconfigurableSlam{})
	_ = resource.Reconfigurable(&reconfigurableSlam{})
	_ = goutils.ContextCloser(&reconfigurableSlam{})
)

// Service describes the functions that are available to the service.
type Service interface {
	Position(context.Context, string, map[string]interface{}) (*referenceframe.PoseInFrame, error)
	GetPosition(context.Context, string) (spatialmath.Pose, string, error)
	GetMap(
		context.Context,
		string,
		string,
		*referenceframe.PoseInFrame,
		bool,
		map[string]interface{},
	) (string, image.Image, *vision.Object, error)
	GetInternalState(ctx context.Context, name string) ([]byte, error)
	GetPointCloudMapStream(ctx context.Context, name string) (func() ([]byte, error), error)
	GetInternalStateStream(ctx context.Context, name string) (func() ([]byte, error), error)
	resource.Generic
}

// Helper function that concatenates the chunks from a streamed grpc endpoint.
func helperConcatenateChunksToFull(f func() ([]byte, error)) ([]byte, error) {
	var fullBytes []byte
	for {
		chunk, err := f()
		if errors.Is(err, io.EOF) {
			return fullBytes, nil
		}
		if err != nil {
			return nil, err
		}

		fullBytes = append(fullBytes, chunk...)
	}
}

// GetPointCloudMapFull concatenates the streaming responses from GetPointCloudMapStream into a full point cloud.
func GetPointCloudMapFull(ctx context.Context, slamSvc Service, name string) ([]byte, error) {
	ctx, span := trace.StartSpan(ctx, "slam::GetPointCloudMapFull")
	defer span.End()
	callback, err := slamSvc.GetPointCloudMapStream(ctx, name)
	if err != nil {
		return nil, err
	}
	return helperConcatenateChunksToFull(callback)
}

// GetInternalStateFull concatenates the streaming responses from GetInternalStateStream into
// the internal serialized state of the slam algorithm.
func GetInternalStateFull(ctx context.Context, slamSvc Service, name string) ([]byte, error) {
	ctx, span := trace.StartSpan(ctx, "slam::GetInternalStateFull")
	defer span.End()
	callback, err := slamSvc.GetInternalStateStream(ctx, name)
	if err != nil {
		return nil, err
	}
	return helperConcatenateChunksToFull(callback)
}

type reconfigurableSlam struct {
	mu     sync.RWMutex
	name   resource.Name
	actual Service
}

func (svc *reconfigurableSlam) Name() resource.Name {
	return svc.name
}

func (svc *reconfigurableSlam) Position(
	ctx context.Context,
	val string,
	extra map[string]interface{},
) (*referenceframe.PoseInFrame, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.Position(ctx, val, extra)
}

func (svc *reconfigurableSlam) GetPosition(
	ctx context.Context,
	val string,
) (spatialmath.Pose, string, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.GetPosition(ctx, val)
}

func (svc *reconfigurableSlam) GetMap(ctx context.Context,
	name string,
	mimeType string,
	cp *referenceframe.PoseInFrame,
	include bool,
	extra map[string]interface{},
) (string, image.Image, *vision.Object, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.GetMap(ctx, name, mimeType, cp, include, extra)
}

func (svc *reconfigurableSlam) GetInternalState(ctx context.Context, name string) ([]byte, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.GetInternalState(ctx, name)
}

func (svc *reconfigurableSlam) GetPointCloudMapStream(ctx context.Context, name string) (func() ([]byte, error), error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.GetPointCloudMapStream(ctx, name)
}

func (svc *reconfigurableSlam) GetInternalStateStream(ctx context.Context, name string) (func() ([]byte, error), error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.GetInternalStateStream(ctx, name)
}

func (svc *reconfigurableSlam) DoCommand(ctx context.Context,
	cmd map[string]interface{},
) (map[string]interface{}, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return svc.actual.DoCommand(ctx, cmd)
}

func (svc *reconfigurableSlam) Close(ctx context.Context) error {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	return goutils.TryClose(ctx, svc.actual)
}

// Reconfigure replaces the old slam service with a new slam.
func (svc *reconfigurableSlam) Reconfigure(ctx context.Context, newSvc resource.Reconfigurable) error {
	svc.mu.Lock()
	defer svc.mu.Unlock()
	rSvc, ok := newSvc.(*reconfigurableSlam)
	if !ok {
		return utils.NewUnexpectedTypeError(svc, newSvc)
	}
	if err := goutils.TryClose(ctx, svc.actual); err != nil {
		golog.Global().Errorw("error closing old", "error", err)
	}
	svc.actual = rSvc.actual
	return nil
}

// WrapWithReconfigurable wraps a slam service as a Reconfigurable.
func WrapWithReconfigurable(s interface{}, name resource.Name) (resource.Reconfigurable, error) {
	svc, ok := s.(Service)
	if !ok {
		return nil, NewUnimplementedInterfaceError(s)
	}

	if reconfigurable, ok := s.(*reconfigurableSlam); ok {
		return reconfigurable, nil
	}

	return &reconfigurableSlam{name: name, actual: svc}, nil
}
