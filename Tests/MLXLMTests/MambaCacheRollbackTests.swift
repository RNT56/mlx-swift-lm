import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// §4: pins the accept→state-index mapping for speculative rollback on the recurrent
/// (MambaCache) hybrid path — the highest-risk off-by-one. Pure CPU, synthetic states,
/// no model. The real on-hybrid correctness is earned by the differential fuzz gate
/// (byte-identical output + cacheStateHash) once the verify-mode scan + iterator wire it.
@Suite
struct MambaCacheRollbackTests {

    private func setState(_ c: MambaCache, _ v: Float) {
        c.state = [MLXArray(v), MLXArray(v * 10)]  // 2 slots: conv, ssm
    }
    private func slot0(_ c: MambaCache) -> Float { c.state[0].item(Float.self) }
    private func slot1(_ c: MambaCache) -> Float { c.state[1].item(Float.self) }

    @Test func supportsSpeculativeRollback() {
        #expect(MambaCache().supportsSpeculativeRollback)
    }

    @Test func selectAcceptedStepCommitsTheIndexedState() {
        let c = MambaCache()
        setState(c, 0)  // pre-verify state
        c.beginVerify()
        #expect(c.inVerify)
        // Scan consumes 4 verify tokens; state after token i is i.
        for i in 1 ... 4 {
            setState(c, Float(i))
            c.recordVerifyStep()
        }
        // Accept 2 committed tokens -> state after the 2nd consumed token.
        c.selectAcceptedStep(consumed: 2)
        #expect(slot0(c) == 2.0)
        #expect(slot1(c) == 20.0)
        #expect(c.offset == 2)  // preOffset(0) + consumed(2)
        #expect(c.inVerify == false)
    }

    @Test func fullAcceptCommitsLastStep() {
        let c = MambaCache()
        setState(c, 0)
        c.beginVerify()
        for i in 1 ... 5 {
            setState(c, Float(i))
            c.recordVerifyStep()
        }
        c.selectAcceptedStep(consumed: 5)
        #expect(slot0(c) == 5.0)
        #expect(c.offset == 5)
    }

    @Test func discardVerifyRestoresPreState() {
        let c = MambaCache()
        setState(c, 7)  // pre-verify state
        c.beginVerify()
        for i in 1 ... 3 {
            setState(c, Float(i))
            c.recordVerifyStep()
        }
        c.discardVerify()
        #expect(slot0(c) == 7.0)
        #expect(slot1(c) == 70.0)
        #expect(c.offset == 0)
        #expect(c.inVerify == false)
    }

    @Test func outOfRangeConsumedEndsVerifyWithoutCorruptingOffset() {
        let c = MambaCache()
        setState(c, 9)
        c.beginVerify()
        for i in 1 ... 2 {
            setState(c, Float(i))
            c.recordVerifyStep()
        }
        c.selectAcceptedStep(consumed: 99)  // invalid (> recorded) -> no commit, ends verify
        #expect(c.inVerify == false)
        #expect(c.offset == 0)  // not advanced on an invalid selection
    }

    @Test func notTrimmableSoStandardSpeculationAdmissionStillExcludesIt() {
        // MambaCache must NOT be trimmable (no per-token KV); speculation on the hybrid
        // must route through selectAcceptedStep, not trimPromptCache.
        #expect(MambaCache().isTrimmable == false)
        #expect(canTrimPromptCache([MambaCache()]) == false)
    }
}
