#pragma once

#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include <memory>
#include <array>
#include <atomic>
#include <GLFW/glfw3.h>

// Set from main() once the MuJoCo viewer window exists (before any joystick
// object is constructed), so KeyboardJoystick can poll key state from it.
inline GLFWwindow* g_glfw_window = nullptr;
inline std::array<std::atomic<bool>, GLFW_KEY_LAST + 1> g_key_pressed{};


class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


// Emulates a Unitree WirelessController from the MuJoCo viewer window's
// keyboard, for when no physical gamepad is available.
//
// Mapping (held-down = pressed):
//   1                  -> F1   (FixStand)
//   2                  -> F2   (Velocity)
//   4                  -> Y    (Mimic)
//   0                  -> back (emergency stop)
//   W/S                -> forward/backward
//   A/D                -> left/right
//   Q/E                -> turn left/right
//   Arrow keys         -> D-pad (up/down/left/right)
//   B / X / Y          -> B / X / Y
//   Left/Right Shift   -> LT / RT
//   Left/Right Ctrl    -> LB / RB
//   Enter              -> start
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    explicit KeyboardJoystick(GLFWwindow* window) : window_(window)
    {
        if (!window_) {
            std::cout << "Error: Keyboard joystick has no window to read from." << std::endl;
            exit(1);
        }
    }

    void update() override
    {
        auto held = [](int key) { return g_key_pressed[key].load(std::memory_order_relaxed); };

        F1(held(GLFW_KEY_1) ? 1 : 0);
        F2(held(GLFW_KEY_2) ? 1 : 0);
        back(held(GLFW_KEY_0) ? 1 : 0);
        start(held(GLFW_KEY_ENTER) ? 1 : 0);
        LB(held(GLFW_KEY_LEFT_CONTROL) ? 1 : 0);
        RB(held(GLFW_KEY_RIGHT_CONTROL) ? 1 : 0);
        A(0);
        B(held(GLFW_KEY_B) ? 1 : 0);
        X(held(GLFW_KEY_X) ? 1 : 0);
        Y(held(GLFW_KEY_4) || held(GLFW_KEY_Y) ? 1 : 0);
        up(held(GLFW_KEY_UP) ? 1 : 0);
        down(held(GLFW_KEY_DOWN) ? 1 : 0);
        left(held(GLFW_KEY_LEFT) ? 1 : 0);
        right(held(GLFW_KEY_RIGHT) ? 1 : 0);
        LT(held(GLFW_KEY_LEFT_SHIFT) ? 1.0f : 0.0f);
        RT(held(GLFW_KEY_RIGHT_SHIFT) ? 1.0f : 0.0f);
        lx(0.5f * (held(GLFW_KEY_D) - held(GLFW_KEY_A)));
        ly(held(GLFW_KEY_W) == held(GLFW_KEY_S) ? 0.0f
           : (held(GLFW_KEY_W) ? 1.0f : -0.5f));
        rx(static_cast<float>(held(GLFW_KEY_E) - held(GLFW_KEY_Q)));
        ry(0.0f);
    }
private:
    GLFWwindow* window_;
};


class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};
