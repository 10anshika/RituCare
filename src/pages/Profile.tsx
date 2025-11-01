import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { User, Settings, Bell, Download, Lock } from "lucide-react";
import { toast } from "sonner";

export default function Profile() {
  const [profile, setProfile] = useState({
    name: "User",
    email: "user@example.com",
    age: "25",
    cycleLength: "28",
    periodLength: "5",
  });

  const [notifications, setNotifications] = useState({
    periodReminder: true,
    fertileWindow: true,
    waterReminder: true,
    dailyTips: false,
  });

  const handleProfileUpdate = () => {
    toast.success("Profile updated successfully");
  };

  const handleExportData = () => {
    toast.success("Data export started", {
      description: "Your data will be downloaded shortly",
    });
  };

  return (
    <div className="min-h-screen bg-[var(--gradient-subtle)]">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl">
            <User className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Profile</h1>
            <p className="text-muted-foreground">Manage your account and preferences</p>
          </div>
        </div>

        {/* Profile Information */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <Settings className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Personal Information</h2>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={profile.name}
                  onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                  className="border-border"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                  className="border-border"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  value={profile.age}
                  onChange={(e) => setProfile({ ...profile, age: e.target.value })}
                  className="border-border"
                />
              </div>
            </div>

            <Button onClick={handleProfileUpdate} className="bg-gradient-to-r from-primary to-primary/80">
              Save Changes
            </Button>
          </div>
        </Card>

        {/* Cycle Settings */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <Settings className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Cycle Settings</h2>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="cycleLength">Average Cycle Length (days)</Label>
                <Input
                  id="cycleLength"
                  type="number"
                  value={profile.cycleLength}
                  onChange={(e) => setProfile({ ...profile, cycleLength: e.target.value })}
                  className="border-border"
                />
                <p className="text-xs text-muted-foreground">Typical range: 21-35 days</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="periodLength">Average Period Length (days)</Label>
                <Input
                  id="periodLength"
                  type="number"
                  value={profile.periodLength}
                  onChange={(e) => setProfile({ ...profile, periodLength: e.target.value })}
                  className="border-border"
                />
                <p className="text-xs text-muted-foreground">Typical range: 3-7 days</p>
              </div>
            </div>

            <Button onClick={handleProfileUpdate} className="bg-gradient-to-r from-primary to-primary/80">
              Update Cycle Settings
            </Button>
          </div>
        </Card>

        {/* Notification Preferences */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <Bell className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Notifications</h2>
          </div>

          <div className="space-y-4">
            {Object.entries(notifications).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between py-3 border-b border-border last:border-0">
                <div>
                  <p className="font-medium text-foreground">
                    {key
                      .replace(/([A-Z])/g, " $1")
                      .replace(/^./, (str) => str.toUpperCase())}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {key === "periodReminder" && "Get notified 2 days before your period"}
                    {key === "fertileWindow" && "Receive alerts for your fertile window"}
                    {key === "waterReminder" && "Daily reminders to stay hydrated"}
                    {key === "dailyTips" && "Get daily health and wellness tips"}
                  </p>
                </div>
                <Switch
                  checked={value}
                  onCheckedChange={(checked) => {
                    setNotifications({ ...notifications, [key]: checked });
                    toast.success(`Notification ${checked ? "enabled" : "disabled"}`);
                  }}
                />
              </div>
            ))}
          </div>
        </Card>

        {/* Data & Privacy */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <Lock className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Data & Privacy</h2>
          </div>

          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-muted/50">
              <p className="text-sm text-muted-foreground mb-4">
                Your health data is private and secure. Export your data at any time.
              </p>
              <Button onClick={handleExportData} variant="outline" className="w-full md:w-auto">
                <Download className="w-4 h-4 mr-2" />
                Export My Data
              </Button>
            </div>

            <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
              <h3 className="font-medium text-foreground mb-2">Delete Account</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Permanently delete your account and all associated data.
              </p>
              <Button variant="destructive" className="w-full md:w-auto">
                Delete Account
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
