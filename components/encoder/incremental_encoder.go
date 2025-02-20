package encoder

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/edaniels/golog"
	"github.com/pkg/errors"
	"go.viam.com/utils"

	"go.viam.com/rdk/components/board"
	"go.viam.com/rdk/components/generic"
	"go.viam.com/rdk/config"
	"go.viam.com/rdk/registry"
	"go.viam.com/rdk/resource"
)

var incrModel = resource.NewDefaultModel("incremental")

func init() {
	registry.RegisterComponent(
		Subtype,
		incrModel,
		registry.Component{Constructor: func(
			ctx context.Context,
			deps registry.Dependencies,
			config config.Component,
			logger golog.Logger,
		) (interface{}, error) {
			return NewIncrementalEncoder(ctx, deps, config, logger)
		}})

	config.RegisterComponentAttributeMapConverter(
		Subtype,
		incrModel,
		func(attributes config.AttributeMap) (interface{}, error) {
			var conf IncrementalConfig
			return config.TransformAttributeMapToStruct(&conf, attributes)
		},
		&IncrementalConfig{})
}

// IncrementalEncoder keeps track of a motor position using a rotary incremental encoder.
type IncrementalEncoder struct {
	A, B     board.DigitalInterrupt
	position int64
	pRaw     int64
	pState   int64

	logger                  golog.Logger
	CancelCtx               context.Context
	cancelFunc              func()
	activeBackgroundWorkers sync.WaitGroup

	generic.Unimplemented
}

// IncrementalPins describes the configuration of Pins for a quadrature encoder.
type IncrementalPins struct {
	A string `json:"a"`
	B string `json:"b"`
}

// IncrementalConfig describes the configuration of a quadrature encoder.
type IncrementalConfig struct {
	Pins      IncrementalPins `json:"pins"`
	BoardName string          `json:"board"`
}

// Validate ensures all parts of the config are valid.
func (config *IncrementalConfig) Validate(path string) ([]string, error) {
	var deps []string

	if config.Pins.A == "" {
		return nil, errors.New("expected nonempty string for a")
	}
	if config.Pins.B == "" {
		return nil, errors.New("expected nonempty string for b")
	}

	if len(config.BoardName) == 0 {
		return nil, errors.New("expected nonempty board")
	}
	deps = append(deps, config.BoardName)

	return deps, nil
}

// NewIncrementalEncoder creates a new IncrementalEncoder.
func NewIncrementalEncoder(
	ctx context.Context,
	deps registry.Dependencies,
	cfg config.Component,
	logger golog.Logger,
) (*IncrementalEncoder, error) {
	cancelCtx, cancelFunc := context.WithCancel(ctx)
	e := &IncrementalEncoder{logger: logger, CancelCtx: cancelCtx, cancelFunc: cancelFunc, position: 0, pRaw: 0, pState: 0}
	if cfg, ok := cfg.ConvertedAttributes.(*IncrementalConfig); ok {
		board, err := board.FromDependencies(deps, cfg.BoardName)
		if err != nil {
			return nil, err
		}

		e.A, ok = board.DigitalInterruptByName(cfg.Pins.A)
		if !ok {
			return nil, errors.Errorf("cannot find pin (%s) for incremental Encoder", cfg.Pins.A)
		}
		e.B, ok = board.DigitalInterruptByName(cfg.Pins.B)
		if !ok {
			return nil, errors.Errorf("cannot find pin (%s) for incremental Encoder", cfg.Pins.B)
		}

		e.Start(ctx)

		return e, nil
	}

	return nil, errors.New("encoder config for incremental Encoder is not valid")
}

// Start starts the IncrementalEncoder background thread.
func (e *IncrementalEncoder) Start(ctx context.Context) {
	/**
	  a rotary encoder looks like

	  picture from https://github.com/joan2937/pigpio/blob/master/EXAMPLES/C/ROTARY_ENCODER/rotary_encoder.c
	    1   2     3    4    1    2    3    4     1

	            +---------+         +---------+      0
	            |         |         |         |
	  A         |         |         |         |
	            |         |         |         |
	  +---------+         +---------+         +----- 1

	      +---------+         +---------+            0
	      |         |         |         |
	  B   |         |         |         |
	      |         |         |         |
	  ----+         +---------+         +---------+  1

	*/

	// State Transition Table
	//     +---------------+----+----+----+----+
	//     | pState/nState | 00 | 01 | 10 | 11 |
	//     +---------------+----+----+----+----+
	//     |       00      | 0  | -1 | +1 | x  |
	//     +---------------+----+----+----+----+
	//     |       01      | +1 | 0  | x  | -1 |
	//     +---------------+----+----+----+----+
	//     |       10      | -1 | x  | 0  | +1 |
	//     +---------------+----+----+----+----+
	//     |       11      | x  | +1 | -1 | 0  |
	//     +---------------+----+----+----+----+
	// 0 -> same state
	// x -> impossible state

	chanA := make(chan board.Tick)
	chanB := make(chan board.Tick)

	e.A.AddCallback(chanA)
	e.B.AddCallback(chanB)

	aLevel, err := e.A.Value(ctx, nil)
	if err != nil {
		utils.Logger.Errorw("error reading a level", "error", err)
	}
	bLevel, err := e.B.Value(ctx, nil)
	if err != nil {
		utils.Logger.Errorw("error reading b level", "error", err)
	}
	e.pState = aLevel | (bLevel << 1)

	e.activeBackgroundWorkers.Add(1)

	utils.ManagedGo(func() {
		defer e.A.RemoveCallback(chanA)
		defer e.B.RemoveCallback(chanB)
		for {
			select {
			case <-e.CancelCtx.Done():
				return
			default:
			}

			var tick board.Tick

			select {
			case <-e.CancelCtx.Done():
				return
			case tick = <-chanA:
				aLevel = 0
				if tick.High {
					aLevel = 1
				}
			case tick = <-chanB:
				bLevel = 0
				if tick.High {
					bLevel = 1
				}
			}
			nState := aLevel | (bLevel << 1)
			if e.pState == nState {
				continue
			}
			switch (e.pState << 2) | nState {
			case 0b0001:
				fallthrough
			case 0b0111:
				fallthrough
			case 0b1000:
				fallthrough
			case 0b1110:
				e.dec()
				atomic.StoreInt64(&e.position, atomic.LoadInt64(&e.pRaw)>>1)
				e.pState = nState
			case 0b0010:
				fallthrough
			case 0b0100:
				fallthrough
			case 0b1011:
				fallthrough
			case 0b1101:
				e.inc()
				atomic.StoreInt64(&e.position, atomic.LoadInt64(&e.pRaw)>>1)
				e.pState = nState
			}
		}
	}, e.activeBackgroundWorkers.Done)
}

// TicksCount returns number of ticks since last zeroing.
func (e *IncrementalEncoder) TicksCount(ctx context.Context, extra map[string]interface{}) (float64, error) {
	res := atomic.LoadInt64(&e.position)
	return float64(res), nil
}

// Reset sets the current position of the motor (adjusted by a given offset)
// to be its new zero position..
func (e *IncrementalEncoder) Reset(ctx context.Context, offset float64, extra map[string]interface{}) error {
	if err := ValidateIntegerOffset(offset); err != nil {
		return err
	}
	offsetInt := int64(offset)
	atomic.StoreInt64(&e.position, offsetInt)
	atomic.StoreInt64(&e.pRaw, (offsetInt<<1)|atomic.LoadInt64(&e.pRaw)&0x1)
	return nil
}

// RawPosition returns the raw position of the encoder.
func (e *IncrementalEncoder) RawPosition() int64 {
	return atomic.LoadInt64(&e.pRaw)
}

func (e *IncrementalEncoder) inc() {
	atomic.AddInt64(&e.pRaw, 1)
}

func (e *IncrementalEncoder) dec() {
	atomic.AddInt64(&e.pRaw, -1)
}

// Close shuts down the IncrementalEncoder.
func (e *IncrementalEncoder) Close() error {
	e.logger.Debug("Closing incremental Encoder")
	e.cancelFunc()
	e.activeBackgroundWorkers.Wait()
	return nil
}
